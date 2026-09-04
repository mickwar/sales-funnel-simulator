"""Tests for schema migration loading/application (PLAN.md section 3).

No live Postgres in this environment, so `init_db` is exercised against a small fake
connection that just records what it was asked to execute -- these tests check that the right
SQL gets loaded and sent, not that Postgres accepts it. Run the migration against a real
instance (POSTGRES_URL) to check that.
"""

from __future__ import annotations

import pytest

from funnel_sim.storage.schema import (
    INITIAL_SCHEMA_MIGRATION,
    init_db,
    load_migration_sql,
)


class _FakeConnection:
    def __init__(self) -> None:
        self.executed: list[str] = []
        self.committed = False

    def execute(self, query: str) -> None:
        self.executed.append(query)

    def commit(self) -> None:
        self.committed = True


def test_load_migration_sql_contains_the_core_tables():
    sql = load_migration_sql()
    for table in (
        "runs",
        "events",
        "accounts_current",
        "leads_current",
        "reps_current",
        "rep_quotas_current",
        "opportunities_current",
        "tasks_current",
    ):
        assert f"CREATE TABLE {table}" in sql, f"missing CREATE TABLE for {table}"


def test_load_migration_sql_indexes_every_current_table_by_run_id():
    sql = load_migration_sql()
    for table in (
        "accounts_current",
        "leads_current",
        "reps_current",
        "rep_quotas_current",
        "opportunities_current",
        "tasks_current",
    ):
        assert f"ON {table} (run_id" in sql, f"missing a run_id-leading index on {table}"


def test_load_migration_sql_raises_a_clear_error_for_an_unknown_migration():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_migration_sql("0099_does_not_exist.sql")


def test_init_db_executes_the_migration_sql_and_commits():
    conn = _FakeConnection()
    init_db(conn)

    assert len(conn.executed) == 1
    assert "CREATE TABLE runs" in conn.executed[0]
    assert conn.committed is True


def test_init_db_uses_the_initial_schema_migration_by_default():
    assert INITIAL_SCHEMA_MIGRATION == "0001_initial_schema.sql"
