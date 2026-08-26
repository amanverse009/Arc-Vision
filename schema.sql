-- Kanoon Wala database schema (PostgreSQL 15+ with pgvector extension)
-- This mirrors app/models.py. Alembic will normally generate/apply this for you
-- (see alembic/), but this file is here for manual setup / review.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TYPE user_role AS ENUM ('citizen', 'admin');
CREATE TYPE source_type AS ENUM ('constitution_article', 'act_section', 'court_case', 'nalsa_resource');
CREATE TYPE document_status AS ENUM ('uploaded', 'processing', 'simplified', 'failed');
CREATE TYPE message_role AS ENUM ('user', 'assistant');

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    phone_number VARCHAR(15) UNIQUE NOT NULL,
    full_name VARCHAR(120),
    preferred_language VARCHAR(10) DEFAULT 'hi',
    role user_role DEFAULT 'citizen',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT now(),
    last_login_at TIMESTAMPTZ
);
CREATE INDEX idx_users_phone ON users(phone_number);

CREATE TABLE otp_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    phone_number VARCHAR(15) NOT NULL,
    otp_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    is_used BOOLEAN DEFAULT FALSE,
    attempt_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_otp_phone ON otp_requests(phone_number);

CREATE TABLE legal_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_type source_type NOT NULL,
    title VARCHAR(300) NOT NULL,
    citation VARCHAR(200),
    full_text TEXT NOT NULL,
    simplified_text TEXT,
    simplified_text_hi TEXT,
    summary_json JSONB,
    source_url VARCHAR(500),
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_legal_sources_type ON legal_sources(source_type);
CREATE INDEX idx_legal_sources_title_trgm ON legal_sources USING gin (title gin_trgm_ops);

CREATE TABLE legal_source_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES legal_sources(id) ON DELETE CASCADE,
    chunk_index INTEGER DEFAULT 0,
    content TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL
);
CREATE INDEX idx_chunks_source ON legal_source_chunks(source_id);
-- Approximate nearest-neighbour index for fast cosine similarity search at scale:
CREATE INDEX idx_chunks_embedding ON legal_source_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_chunks_fts ON legal_source_chunks USING gin (to_tsvector('english', content));

CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(200),
    input_mode VARCHAR(10) DEFAULT 'text',
    language VARCHAR(10) DEFAULT 'hi',
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_chat_sessions_user ON chat_sessions(user_id);

CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role message_role NOT NULL,
    content TEXT NOT NULL,
    audio_url VARCHAR(500),
    cited_source_ids JSONB,
    confidence VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_chat_messages_session ON chat_messages(session_id);

CREATE TABLE user_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    original_filename VARCHAR(300) NOT NULL,
    storage_path VARCHAR(500) NOT NULL,
    extracted_text TEXT,
    simplified_summary TEXT,
    status document_status DEFAULT 'uploaded',
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_documents_user ON user_documents(user_id);

CREATE TABLE admin_audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    actor_user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    event_type VARCHAR(100) NOT NULL,
    event_meta JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_audit_logs_event_type ON admin_audit_logs(event_type);
CREATE INDEX idx_audit_logs_created ON admin_audit_logs(created_at);
