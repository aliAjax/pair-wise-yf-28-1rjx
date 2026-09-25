import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from app import BusinessError, RandomizationStore


class RandomizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = RandomizationStore(Path(self.tmp.name) / "test.db")
        self.store.seed()
        self.trial = self.store.create_trial(
            "coord", "多中心降压研究", "v1.0", ["A", "B"], ["risk"], 4, "seed-2026-001"
        )
        self.store.start_trial("coord", self.trial["id"])

    def tearDown(self):
        self.tmp.cleanup()

    def test_stratified_block_randomization_and_two_person_unblinding(self):
        participants = [
            self.store.enroll("site1", self.trial["id"], f"S001-{i:03d}", {"risk": "low"})
            for i in range(1, 5)
        ]
        self.assertNotIn("arm", participants[0])
        with self.store.connect() as conn:
            arms = [r["arm"] for r in conn.execute(
                "SELECT a.arm FROM allocations a JOIN participants p ON p.allocation_id=a.id WHERE p.trial_id=? ORDER BY p.id",
                (self.trial["id"],),
            ).fetchall()]
        self.assertEqual(Counter(arms), Counter({"A": 2, "B": 2}))
        request = self.store.request_unblinding("site1", participants[0]["id"], "受试者发生严重不良事件需要紧急处理")
        first = self.store.approve_unblinding("monitor1", request["id"])
        self.assertEqual(first["status"], "pending")
        with self.assertRaises(BusinessError) as ctx:
            self.store.approve_unblinding("monitor1", request["id"])
        self.assertEqual(ctx.exception.code, "distinct_approver_required")
        second = self.store.approve_unblinding("monitor2", request["id"])
        self.assertEqual(second["status"], "approved")
        self.assertIn(second["arm"], {"A", "B"})

    def test_idempotent_enrollment_site_isolation_and_protocol_lock(self):
        first = self.store.enroll("site1", self.trial["id"], "S001-001", {"risk": "high"})
        again = self.store.enroll("site1", self.trial["id"], "S001-001", {"risk": "high"})
        self.assertEqual(first["id"], again["id"])
        self.assertTrue(again["idempotent"])
        with self.store.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM participants").fetchone()[0], 1)
        with self.assertRaises(BusinessError) as ctx:
            self.store.get_participant("site2", first["id"])
        self.assertEqual(ctx.exception.code, "site_isolation")
        with self.assertRaises(BusinessError) as ctx:
            self.store.update_protocol("coord", self.trial["id"], "v2")
        self.assertEqual(ctx.exception.code, "protocol_locked")

    def test_site_suspend_blocks_enrollment_but_keeps_existing_and_unblinding(self):
        p1 = self.store.enroll("site1", self.trial["id"], "S001-001", {"risk": "low"})
        p2 = self.store.enroll("site1", self.trial["id"], "S001-002", {"risk": "low"})
        with self.assertRaises(BusinessError) as ctx:
            self.store.suspend_site("site1", self.trial["id"], "S001", "监查发现方案偏离")
        self.assertEqual(ctx.exception.code, "forbidden")
        with self.assertRaises(BusinessError) as ctx:
            self.store.suspend_site("coord", self.trial["id"], "S001", "短")
        self.assertEqual(ctx.exception.code, "reason_required")
        with self.assertRaises(BusinessError) as ctx:
            self.store.suspend_site("coord", self.trial["id"], "S999", "监查发现方案偏离")
        self.assertEqual(ctx.exception.code, "unknown_site")
        result = self.store.suspend_site("coord", self.trial["id"], "S001", "监查发现方案偏离需整改")
        self.assertEqual(result["status"], "suspended")
        with self.assertRaises(BusinessError) as ctx:
            self.store.suspend_site("coord", self.trial["id"], "S001", "重复暂停原因说明")
        self.assertEqual(ctx.exception.code, "already_suspended")
        with self.assertRaises(BusinessError) as ctx:
            self.store.enroll("site1", self.trial["id"], "S001-003", {"risk": "low"})
        self.assertEqual(ctx.exception.code, "site_suspended")
        replay = self.store.enroll("site1", self.trial["id"], "S001-001", {"risk": "low"})
        self.assertTrue(replay["idempotent"])
        self.assertEqual(replay["id"], p1["id"])
        other = self.store.enroll("site2", self.trial["id"], "S002-001", {"risk": "low"})
        self.assertFalse(other["idempotent"])
        self.assertEqual(self.store.get_participant("site1", p2["id"])["id"], p2["id"])
        request = self.store.request_unblinding("site1", p1["id"], "严重不良事件需紧急揭盲处理")
        self.store.approve_unblinding("monitor1", request["id"])
        done = self.store.approve_unblinding("monitor2", request["id"])
        self.assertEqual(done["status"], "approved")
        self.assertIn(done["arm"], {"A", "B"})

    def test_site_resume_continues_stratum_sequence_without_reshuffle(self):
        enrolled = [
            self.store.enroll("site1", self.trial["id"], f"S001-{i:03d}", {"risk": "low"})
            for i in range(1, 3)
        ]
        self.store.suspend_site("coord", self.trial["id"], "S001", "设备校准超期暂停入组")
        with self.assertRaises(BusinessError) as ctx:
            self.store.resume_site("site2", self.trial["id"], "S001")
        self.assertEqual(ctx.exception.code, "forbidden")
        resumed = self.store.resume_site("coord", self.trial["id"], "S001", "整改完成恢复入组")
        self.assertEqual(resumed["status"], "active")
        with self.assertRaises(BusinessError) as ctx:
            self.store.resume_site("coord", self.trial["id"], "S001")
        self.assertEqual(ctx.exception.code, "not_suspended")
        enrolled.append(self.store.enroll("site1", self.trial["id"], "S001-003", {"risk": "low"}))
        with self.store.connect() as conn:
            rows = conn.execute(
                """SELECT a.sequence,a.used_by,a.arm FROM allocations a
                   JOIN participants p ON p.allocation_id=a.id
                   WHERE p.trial_id=? AND p.site_id='S001' ORDER BY p.id""",
                (self.trial["id"],),
            ).fetchall()
        self.assertEqual([r["sequence"] for r in rows], [1, 2, 3])
        self.assertEqual([r["used_by"] for r in rows], [p["id"] for p in enrolled])

    def test_site_status_listing_and_summary_stay_blinded(self):
        self.store.enroll("site1", self.trial["id"], "S001-001", {"risk": "high"})
        self.store.suspend_site("coord", self.trial["id"], "S001", "设备校准超期暂停入组")
        view = self.store.list_site_statuses("coord", self.trial["id"])
        by_site = {s["site_id"]: s for s in view["sites"]}
        self.assertEqual(by_site["S001"]["status"], "suspended")
        self.assertEqual(by_site["S001"]["reason"], "设备校准超期暂停入组")
        self.assertEqual(by_site["S001"]["enrolled"], 1)
        self.assertEqual(by_site["S002"]["status"], "active")
        self.assertEqual([h["action"] for h in by_site["S001"]["history"]], ["suspend"])
        self.store.resume_site("coord", self.trial["id"], "S001", "整改完成")
        view = self.store.list_site_statuses("monitor1", self.trial["id"])
        s001 = {s["site_id"]: s for s in view["sites"]}["S001"]
        self.assertEqual(s001["status"], "active")
        self.assertIsNone(s001["reason"])
        self.assertEqual([h["action"] for h in s001["history"]], ["suspend", "resume"])
        with self.assertRaises(BusinessError) as ctx:
            self.store.list_site_statuses("site1", self.trial["id"])
        self.assertEqual(ctx.exception.code, "forbidden")
        summary = self.store.trial_summary("coord", self.trial["id"])
        self.assertIn("sites", summary)
        self.assertNotIn('"arm"', json.dumps(summary, ensure_ascii=False))
        draft = self.store.create_trial("coord", "另一项研究", "v1", ["A", "B"], ["risk"], 2, "seed-2026-002")
        with self.assertRaises(BusinessError) as ctx:
            self.store.suspend_site("coord", draft["id"], "S001", "草稿试验不能暂停中心")
        self.assertEqual(ctx.exception.code, "trial_not_running")


if __name__ == "__main__":
    unittest.main()
