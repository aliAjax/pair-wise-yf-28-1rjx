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

    def test_site_suspend_blocks_new_enrollment_but_existing_flows_continue(self):
        p1 = self.store.enroll("site1", self.trial["id"], "S001-201", {"risk": "low"})
        with self.assertRaises(BusinessError) as ctx:
            self.store.suspend_site("coord", self.trial["id"], "S001", "短")
        self.assertEqual(ctx.exception.code, "reason_required")
        with self.assertRaises(BusinessError) as ctx:
            self.store.suspend_site("site1", self.trial["id"], "S001", "发现方案违背需要整改")
        self.assertEqual(ctx.exception.code, "forbidden")
        with self.assertRaises(BusinessError) as ctx:
            self.store.suspend_site("coord", self.trial["id"], "S999", "不存在的中心")
        self.assertEqual(ctx.exception.code, "unknown_site")
        result = self.store.suspend_site("coord", self.trial["id"], "S001", "监查发现方案违背，现场整改")
        self.assertEqual(result["status"], "suspended")
        with self.assertRaises(BusinessError) as ctx:
            self.store.suspend_site("coord", self.trial["id"], "S001", "重复暂停")
        self.assertEqual(ctx.exception.code, "invalid_status")
        with self.assertRaises(BusinessError) as ctx:
            self.store.enroll("site1", self.trial["id"], "S001-202", {"risk": "low"})
        self.assertEqual(ctx.exception.code, "site_suspended")
        again = self.store.enroll("site1", self.trial["id"], "S001-201", {"risk": "low"})
        self.assertEqual(again["id"], p1["id"])
        self.assertTrue(again["idempotent"])
        request = self.store.request_unblinding("site1", p1["id"], "严重不良事件需要紧急揭盲")
        self.store.approve_unblinding("monitor1", request["id"])
        done = self.store.approve_unblinding("monitor2", request["id"])
        self.assertEqual(done["status"], "approved")
        other = self.store.enroll("site2", self.trial["id"], "S002-201", {"risk": "low"})
        self.assertFalse(other["idempotent"])

    def test_resume_continues_stratum_sequence_and_records_history(self):
        for i in range(1, 3):
            self.store.enroll("site1", self.trial["id"], f"S001-30{i}", {"risk": "high"})
        used_sql = (
            "SELECT a.id,a.sequence,a.arm,a.used_by FROM allocations a"
            " JOIN participants p ON p.allocation_id=a.id WHERE p.trial_id=? ORDER BY a.sequence"
        )
        with self.store.connect() as conn:
            before = [dict(r) for r in conn.execute(used_sql, (self.trial["id"],)).fetchall()]
        self.store.suspend_site("coord", self.trial["id"], "S001", "设备校准暂停入组")
        with self.assertRaises(BusinessError) as ctx:
            self.store.resume_site("site2", self.trial["id"], "S001")
        self.assertEqual(ctx.exception.code, "forbidden")
        self.store.resume_site("coord", self.trial["id"], "S001", "整改完成恢复入组")
        self.store.enroll("site1", self.trial["id"], "S001-303", {"risk": "high"})
        with self.store.connect() as conn:
            after = [dict(r) for r in conn.execute(used_sql, (self.trial["id"],)).fetchall()]
        self.assertEqual(after[:2], before)
        self.assertEqual(after[2]["sequence"], before[-1]["sequence"] + 1)
        self.assertNotIn(after[2]["id"], [r["id"] for r in before])
        view = self.store.list_site_statuses("coord", self.trial["id"])
        s001 = next(s for s in view["sites"] if s["site_id"] == "S001")
        self.assertEqual(s001["status"], "active")
        self.assertEqual(s001["enrolled"], 3)
        self.assertEqual([(e["action"], e["actor_id"]) for e in view["events"]], [("suspend", "coord"), ("resume", "coord")])
        self.assertNotIn('"arm"', json.dumps(view, ensure_ascii=False))
        summary = self.store.trial_summary("coord", self.trial["id"])
        self.assertNotIn('"arm"', json.dumps(summary, ensure_ascii=False))
        self.store.suspend_site("coord", self.trial["id"], "S002", "人员培训不到位")
        view2 = self.store.list_site_statuses("monitor1", self.trial["id"])
        s002 = next(s for s in view2["sites"] if s["site_id"] == "S002")
        self.assertEqual(s002["status"], "suspended")
        self.assertEqual(s002["reason"], "人员培训不到位")
        with self.assertRaises(BusinessError) as ctx:
            self.store.list_site_statuses("site1", self.trial["id"])
        self.assertEqual(ctx.exception.code, "forbidden")
        with self.assertRaises(BusinessError) as ctx:
            self.store.resume_site("coord", self.trial["id"], "S001")
        self.assertEqual(ctx.exception.code, "invalid_status")


if __name__ == "__main__":
    unittest.main()
