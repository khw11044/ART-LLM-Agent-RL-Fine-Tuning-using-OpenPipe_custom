import os
import random
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from textwrap import dedent
from typing import List, Literal, Optional

from datasets import Dataset, Features, Sequence, Value, load_dataset
from pydantic import BaseModel, Field
from tqdm import tqdm


from utils.database_schema import SQL_CREATE_TABLES, SQL_CREATE_INDEXES_TRIGGERS

# Database configuration
DB_PATH = "./enron_emails.db"
EMAIL_DATASET_REPO_ID = "corbt/enron-emails"        # 허깅페이스에 있는 데이터셋 레포 아이디 : llm-agent가 검색하고 이메일을 찾을 주체 
SCENARIO_DATASET_REPO_ID = "corbt/enron_emails_sample_questions"    # llm-agent를 학습시키기 위한 질문 샘플들

# Global database connection
db_conn = None

expected_features = Features(
        {
            "message_id": Value("string"),
            "subject": Value("string"),
            "from": Value("string"),
            "to": Sequence(Value("string")),
            "cc": Sequence(Value("string")),
            "bcc": Sequence(Value("string")),
            "date": Value("timestamp[us]"),
            "body": Value("string"),
            "file_name": Value("string"),
        }
    )

# 허깅페이스 데이터셋을 로드해서 sqlite3로 db를 만들고 sqlite3 db를 접근하며 옳은 대답을 해주는 LLM Agent를 만들것임
def create_email_database(EMAIL_DATASET_REPO_ID, DB_PATH):
    """Create the email database from Hugging Face dataset"""
    
    print("Creating email database from Hugging Face dataset...")
    print("This will download and process the full Enron email dataset - this may take several minutes...")
    
    
    # 1. Create database
    # 빈 sql lite 데이터셋을 먼저 만들어줍니다.
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_TABLES)
    conn.commit()
    
    # 2. Load dataset
    print(f"허깅페이스에서 {EMAIL_DATASET_REPO_ID} 데이터셋 로드 중 .....")
    dataset = load_dataset(
        EMAIL_DATASET_REPO_ID, features=expected_features, split="train"
    )
    print(f"Dataset contains {len(dataset)} total emails")
    
    
    # 3. Populate database with ALL emails (not limited to 1000)
    print("Populating database with all emails...")
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")
    conn.execute("BEGIN TRANSACTION;")
    
    # 4. 전처리 
    record_count = 0
    skipped_count = 0
    duplicate_count = 0
    processed_emails = set()  # Track (subject, body, from) tuples for deduplication

    for email_data in tqdm(dataset, desc="Inserting emails"):
        message_id = email_data["message_id"]
        subject = email_data["subject"]
        from_address = email_data["from"]
        date_obj: datetime = email_data["date"]
        body = email_data["body"]
        file_name = email_data["file_name"]
        to_list = [str(addr) for addr in email_data["to"] if addr]
        cc_list = [str(addr) for addr in email_data["cc"] if addr]
        bcc_list = [str(addr) for addr in email_data["bcc"] if addr]

        # Apply the same filters as the original project
        total_recipients = len(to_list) + len(cc_list) + len(bcc_list)

        # Filter out very long emails and those with too many recipients
        if len(body) > 5000:
            skipped_count += 1
            continue

        if total_recipients > 30:
            skipped_count += 1
            continue

        # Deduplication check (same as original project)
        email_key = (subject, body, from_address)
        if email_key in processed_emails:
            duplicate_count += 1
            continue
        else:
            processed_emails.add(email_key)

        date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            """
            INSERT INTO emails (message_id, subject, from_address, date, body, file_name)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (message_id, subject, from_address, date_str, body, file_name),
        )

        # Insert recipients
        recipient_data = []
        for addr in to_list:
            recipient_data.append((message_id, addr, "to"))
        for addr in cc_list:
            recipient_data.append((message_id, addr, "cc"))
        for addr in bcc_list:
            recipient_data.append((message_id, addr, "bcc"))

        if recipient_data:
            cursor.executemany(
                """
                INSERT INTO recipients (email_id, recipient_address, recipient_type)
                VALUES (?, ?, ?)
            """,
                recipient_data,
            )

        record_count += 1

    conn.commit()

    # Create indexes and triggers
    print("Creating indexes and FTS...")
    cursor.executescript(SQL_CREATE_INDEXES_TRIGGERS)
    cursor.execute('INSERT INTO emails_fts(emails_fts) VALUES("rebuild")')
    conn.commit()

    print(f"Successfully created database with {record_count} emails.")
    print(f"Skipped {skipped_count} emails due to length/recipient limits.")
    print(f"Skipped {duplicate_count} duplicate emails.")
    return conn


if __name__=="__main__":
    
    
    # Database configuration
    DB_PATH = "./enron_emails.db"
    EMAIL_DATASET_REPO_ID = "corbt/enron-emails"        # 허깅페이스에 있는 데이터셋 레포 아이디 

    create_email_database(EMAIL_DATASET_REPO_ID, DB_PATH)


