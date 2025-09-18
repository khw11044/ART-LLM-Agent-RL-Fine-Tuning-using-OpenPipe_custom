
from tasks.email.model import Email, SearchResult
import sqlite3


from typing import List, Literal, Optional

# Global database connection
db_conn = None

def get_db_connection(DB_PATH):
    """Get database connection"""
    print(f"Loading existing database from {DB_PATH}")
    db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    return db_conn


def search_emails(
    inbox: str,
    keywords: List[str],
    from_addr: Optional[str] = None,
    to_addr: Optional[str] = None,
    sent_after: Optional[str] = None,
    sent_before: Optional[str] = None,
    max_results: int = 10,
    db_path: Optional[str] = "./enron_emails.db",
) -> List[SearchResult]:
    """Search the email database based on keywords and filters"""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    where_clauses: List[str] = []
    params: List[str | int] = []

    if not keywords:
        raise ValueError("No keywords provided for search.")

    if max_results > 10:
        raise ValueError("max_results must be less than or equal to 10.")

    # FTS5 default is AND, so just join keywords. Escape quotes for safety.
    fts_query = " ".join(f""" "{k.replace('"', '""')}" """ for k in keywords)
    where_clauses.append("fts.emails_fts MATCH ?")
    params.append(fts_query)

    # Inbox filter
    where_clauses.append("""
        (e.from_address = ? OR EXISTS (
            SELECT 1 FROM recipients r_inbox
            WHERE r_inbox.recipient_address = ? AND r_inbox.email_id = e.message_id
        ))
    """)
    params.extend([inbox, inbox])

    if from_addr:
        where_clauses.append("e.from_address = ?")
        params.append(from_addr)

    if to_addr:
        where_clauses.append("""
            EXISTS (
                SELECT 1 FROM recipients r_to
                WHERE r_to.recipient_address = ? AND r_to.email_id = e.message_id
            )
        """)
        params.append(to_addr)

    if sent_after:
        where_clauses.append("e.date >= ?")
        params.append(f"{sent_after} 00:00:00")

    if sent_before:
        where_clauses.append("e.date < ?")
        params.append(f"{sent_before} 00:00:00")

    sql = f"""
        SELECT
            e.message_id,
            snippet(emails_fts, -1, '<b>', '</b>', ' ... ', 15) as snippet
        FROM
            emails e JOIN emails_fts fts ON e.id = fts.rowid
        WHERE
            {" AND ".join(where_clauses)}
        ORDER BY
            e.date DESC
        LIMIT ?;
    """
    params.append(max_results)

    cursor.execute(sql, params)
    results = cursor.fetchall()

    return [SearchResult(message_id=row[0], snippet=row[1]) for row in results]


def read_email(
    message_id: str,
    db_path: Optional[str] = "./enron_emails.db",
) -> Optional[Email]:
    
    """Retrieve a single email by its message_id"""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Get email details
    cursor.execute(
        "SELECT message_id, date, subject, from_address, body, file_name FROM emails WHERE message_id = ?",
        (message_id,),
    )
    email_row = cursor.fetchone()

    if not email_row:
        return None

    msg_id, date, subject, from_addr, body, file_name = email_row

    # Get recipients
    cursor.execute(
        "SELECT recipient_address, recipient_type FROM recipients WHERE email_id = ?",
        (message_id,),
    )
    recipient_rows = cursor.fetchall()

    to_addresses = []
    cc_addresses = []
    bcc_addresses = []

    for addr, type_val in recipient_rows:
        if type_val.lower() == "to":
            to_addresses.append(addr)
        elif type_val.lower() == "cc":
            cc_addresses.append(addr)
        elif type_val.lower() == "bcc":
            bcc_addresses.append(addr)

    return Email(
        message_id=msg_id,
        date=date,
        subject=subject,
        from_address=from_addr,
        to_addresses=to_addresses,
        cc_addresses=cc_addresses,
        bcc_addresses=bcc_addresses,
        body=body,
        file_name=file_name,
    )
