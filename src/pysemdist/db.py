from contextlib import contextmanager
from psycopg2.pool import SimpleConnectionPool
import psycopg2, psycopg2.extras
from .config import settings

_pool = SimpleConnectionPool(
    minconn=settings.db_pool_min,
    maxconn=settings.db_pool_max,
    host=settings.db_host,
    port=settings.db_port,
    dbname=settings.db_name,
    user=settings.db_user,
    password=settings.db_password,
    connect_timeout=settings.db_connect_timeout,
    cursor_factory=psycopg2.extras.DictCursor,
)

def _prepare_conn(conn):
    # Apply per-connection statement timeout (milliseconds)
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = {int(settings.db_statement_timeout_ms)};")
    conn.commit()

@contextmanager
def get_conn():
    conn = _pool.getconn()
    try:
        _prepare_conn(conn)
        yield conn
    finally:
        _pool.putconn(conn)
