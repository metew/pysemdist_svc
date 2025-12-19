from typing import Optional
import pandas as pd
from .db import get_conn

def fetch_country_petitions(start_date: str, end_date: Optional[str], country_code: str, limit: int = 500) -> pd.DataFrame:
    q = '''
    select
        e.id,
        'country_petitions_' || l.id::varchar || '_' || lower(replace(l.country_code,' ','')) as category,
        e.title || left(e.description, 200) as text
    from service_dbs.events e
    inner join service_dbs.locations l on (l.id = e.relevant_location_id)
    where e.total_signature_count > 5
      and e.created_at >= %s
      and (%s is null or e.created_at <= %s)
      and l.country_code = %s
      and e.deleted_at is null
    group by 1,2,3
    limit %s
    '''
    with get_conn() as conn:
        df = pd.read_sql(q, conn, params=[start_date, end_date, end_date, country_code, limit])
    return df

def fetch_city_petitions(start_date: str, end_date: Optional[str], country_code: str, city: str, limit: int = 500) -> pd.DataFrame:
    q = '''
    select
        e.id,
        'city_petitions_' || l.id::varchar || '_' || lower(replace(l.city,' ','')) as category,
        e.title || left(e.description, 200) as text
    from service_dbs.events e
    inner join service_dbs.locations l on (l.id = e.relevant_location_id)
    where e.total_signature_count > 5
      and e.created_at >= %s
      and (%s is null or e.created_at <= %s)
      and l.country_code = %s
      and l.city = %s
      and e.deleted_at is null
    group by 1,2,3
    limit %s
    '''
    with get_conn() as conn:
        df = pd.read_sql(q, conn, params=[start_date, end_date, end_date, country_code, city, limit])
    return df

def fetch_dm_petitions(start_date: str, end_date: Optional[str], country_code: str, dm_id: int, limit: int = 500) -> pd.DataFrame:
    q = '''
    select
        e.id,
        'dm_petitions_' || pt.id::varchar || '_' || lower(replace(pt.name,' ','')) as category,
        e.title || left(e.description, 200) as text
    from service_dbs.events e
    inner join service_dbs.locations l ON (l.id = e.relevant_location_id)
    inner join service_dbs.petitions_petition_targets ppt ON (e.id = ppt.petition_id)
    inner join service_dbs.petition_targets pt ON (ppt.petition_target_id = pt.id)
    inner join service_dbs.offices o
        ON (o.politician_id = pt.targetable_id
        and pt.targetable_type = 'Politician')
    where e.total_signature_count > 5
      and e.created_at >= %s
      and (%s is null or e.created_at <= %s)
      and l.country_code = %s
      and e.deleted_at is null
      and pt.id = %s
      and o.active = 1
      and o.normalized_level = 'city'
    group by 1,2,3
    limit %s
    '''
    with get_conn() as conn:
        df = pd.read_sql(q, conn, params=[start_date, end_date, end_date, country_code, dm_id, limit])
    return df

def fetch_topic_petitions(start_date: str, end_date: Optional[str], country_code: str, topic_id: int, limit: int = 500) -> pd.DataFrame:
    q = '''
    select
        e.id,
        'topic_petitions_' || t.id::varchar || '_' || lower(replace(t.name,' ','')) as category,
        e.title || left(e.description, 200) as text
    from service_dbs.events e
    inner join service_dbs.locations l ON (l.id = e.relevant_location_id)
    inner join service_dbs.taggings p2t on e.id = p2t.taggable_id and taggable_type = 'Event'
    inner join service_dbs.tags as t on t.id = p2t.tag_id
    where e.total_signature_count > 5
      and e.created_at >= %s
      and (%s is null or e.created_at <= %s)
      and l.country_code = %s
      and t.id = %s
    group by 1,2,3
    limit %s
    '''
    with get_conn() as conn:
        df = pd.read_sql(q, conn, params=[start_date, end_date, end_date, country_code, topic_id, limit])
    return df
