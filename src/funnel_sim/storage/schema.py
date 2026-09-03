"""Applies the Postgres schema migration(s) in migrations/ (PLAN.md section 3's event-sourcing
design: append-only `events` plus `<entity>_current` projection tables).

This is the Phase 1 starting point for the storage layer. Bulk COPY loading of generated
entities and DuckDB analytics access (PLAN.md section 5) are the next increment -- this module
only gets a fresh database to the point of having the right tables.

There's no live Postgres in this development sandbox to run integration tests against, so
`init_db` is exercised here with a fake connection object (tests/test_storage/test_schema.py)
that just records the SQL it was asked to execute; the DDL itself is deliberately kept boring
and reviewable-as-SQL (migrations/0001_initial_schema.sql) rather than built up through an ORM
that would hide what's actually being sent to the database. Run it for real against a Postgres
instance from POSTGRES_URL (see .env.example) once one is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"
INITIAL_SCHEMA_MIGRATION = "0001_initial_schema.sql"


class ExecutableConnection(Protocol):
    """The minimal psycopg-connection-shaped interface `init_db` needs -- narrow on purpose so
    tests can pass a lightweight fake instead of a real psycopg connection.
    """

    def execute(self, query: str) -> object: ...

    def commit(self) -> None: ...


def load_migration_sql(migration: str = INITIAL_SCHEMA_MIGRATION) -> str:
    """Read a migration file's SQL text from migrations/."""
    path = MIGRATIONS_DIR / migration
    if not path.is_file():
        raise FileNotFoundError(
            f"Migration {migration!r} not found in {MIGRATIONS_DIR}. "
            "Migrations are loaded from the repo's migrations/ directory, not packaged with "
            "funnel_sim -- this only works when running from a checkout of the repo."
        )
    return path.read_text()


def init_db(conn: ExecutableConnection, migration: str = INITIAL_SCHEMA_MIGRATION) -> None:
    """Apply a schema migration to an open connection and commit it.

    Only meant for a fresh, empty database right now -- running it twice against the same
    database will fail on the second run (no `CREATE TABLE IF NOT EXISTS`, deliberately, so a
    schema drift doesn't get silently papered over). A real migration runner with tracked,
    incremental migrations is future work, not needed yet at solo-use/Phase-1 scale.
    """
    sql = load_migration_sql(migration)
    conn.execute(sql)
    conn.commit()
