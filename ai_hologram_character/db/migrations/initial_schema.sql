-- 用户基础表
-- 用户多维特详细征表（通过交互记录更新）
CREATE TABLE UserProfile (
    id TEXT PRIMARY KEY,                      -- 特征记录唯一标识符
    voice_print_id TEXT,                      -- 用户声纹ID
    face_print_id TEXT,                       -- 用户人脸ID
    fingerprint_id TEXT,                      -- 用户指纹ID
    -- 基本信息1
    external_info_id TEXT,                    -- 关联基本特征信息ID
    -- 基本信息2
    internal_info_id TEXT,                    -- 关联内部特征信息ID
    last_updated TEXT NOT NULL                -- 最后更新时间（ISO8601格式）
);

-- 用户基本信息表
CREATE TABLE ExternalInfo (
    id TEXT PRIMARY KEY,                      -- 用户唯一标识符
    user_id TEXT NOT NULL,                    -- 用户ID
    -- 基本信息（姓名，年龄，出生，国籍，民族，籍贯）
    nickname TEXT,                            -- 用户昵称
    name TEXT,                                -- 用户姓名
    avatar_url TEXT,                          -- 用户头像URL
    gender TEXT,                              -- 用户性别
    age INTEGER,                              -- 用户年龄
    birth_date TEXT,                          -- 出生日期（YYYY-MM-DD格式）
    telephone TEXT,                           -- 联系电话
    address TEXT,                             -- 联系地址
    mail TEXT,                                -- 电子邮箱
    nationality TEXT,                         -- 国籍
    ethnicity TEXT,                           -- 民族
    birthplace TEXT,                          -- 籍贯
    education TEXT,                           -- 教育程度
    marry_status TEXT,                        -- 婚姻状况
    -- 政治身份
    political_status TEXT,                    -- 政治身份
    -- 背景信息（教育背景，职业经历，家庭状况，社会地位）
    identity_background TEXT,                 -- 背景概述
    career_summary TEXT,                      -- 职业经历
    family_status TEXT,                       -- 家庭状况
    social_status TEXT                        -- 社会地位
);

-- 用户内部信息表
CREATE TABLE InternalInfo (
    id TEXT PRIMARY KEY,                      -- 用户唯一标识符
    user_id TEXT NOT NULL,                    -- 用户ID
    -- 外在特征
    physical_features TEXT,                   -- 体貌特征（包含外貌特征[身高体重等]，穿着打扮，仪表仪容等）
    expression_style TEXT,                    -- 表达方式（包含语言表达，肢体语言，面部表情等）
    social_behavior TEXT,                     -- 社交行为（包含社交技巧，人际交往方式，社交圈子等）
    -- 内在特征
    cognitive_traits TEXT,                    -- 认知特征（包含智力，知识储备，思维方式，创造力等）
    emotional_traits TEXT,                    -- 情感特征（包含情绪稳定性，敏感度等）
    -- 用户人格特征
    self_perception TEXT,                     -- 自我认知
    identity_aspects TEXT,                    -- 身份认同方面
    life_goals TEXT,                          -- 人生目标
    value_system TEXT,                        -- 价值观体系（包含道德观念，信仰体系等）
    personality_core TEXT,                    -- 人格核心特质（包含内在的核心特质，生活态度，优势与短板等）
    -- 健康状况
    health_status TEXT                        -- 健康状况
);

-- 用户关系表（记录用户之间的关系）
CREATE TABLE UserRelationship (
    id TEXT PRIMARY KEY,                      -- 关系唯一标识符
    user_id TEXT,                             -- 用户ID
    target_user_id TEXT,                      -- 目标用户ID
    relationship_type TEXT,                   -- （亲人（具体称谓）/恋人/朋友/同事等） 如果有多个身份，可以"同事，朋友"这样定义
    strength REAL,                            -- 关系强度（0-1）
    last_updated TEXT NOT NULL                -- 最后更新时间（ISO8601格式）
);

-- 隐私信息表（通过交互记录更新）
CREATE TABLE Secret (
    id TEXT PRIMARY KEY,                      -- 隐私信息唯一标识符
    user_id TEXT,                             -- 用户ID
    secret_type TEXT NOT NULL,                -- 隐私信息类型（个人/家庭/财务等）
    content TEXT NOT NULL,                    -- 隐私信息内容
    last_updated TEXT NOT NULL                -- 最后更新时间（ISO8601格式）
);

-- 情感偏好和交互习惯表（通过交互记录更新）
CREATE TABLE EmotionPreference (
    id TEXT PRIMARY KEY,                      -- 情感偏好唯一标识符
    user_id TEXT,                             -- 用户ID
    emotion_type TEXT,                        -- 情感偏好
    interaction_habits TEXT,                  -- 情感强度（0-1）
    last_updated TEXT                         -- 最后更新时间（ISO8601格式）
);

-- 交互记录表
CREATE TABLE Interaction (
    id TEXT PRIMARY KEY,                      -- 交互唯一标识符
    memory_id TEXT,                           -- 关联记忆ID[1对1]
    user_intent TEXT,                         -- 用户意图
    context TEXT,                             -- 交互上下文信息[梗概内容]
    -- 交互环境信息，记录交互发生的环境上下文，增强情境感知能力
    location_type TEXT,                       -- 位置类型（家/办公室/公共场所等）
    ambient_factors TEXT,                     -- 环境因素
    created_at TEXT NOT NULL                  -- 创建时间（ISO8601格式）
);

-- 记忆表（未压缩 支持多种记忆类型和检索）
CREATE TABLE Memory (
    id TEXT PRIMARY KEY,                      -- 记忆唯一标识符
    summary TEXT,                             -- 交互上下文信息[梗概内容]
    content_text TEXT,                        -- 记忆内容（详细对话文本）
    milvus_id TEXT,                           -- 记忆向量ID
    embedding_dimensions TEXT,                -- 对话整体向量嵌入
    importance_score REAL,                    -- 记忆重要性评分（0-1）
    created_at TEXT NOT NULL                  -- 记忆创建时间（ISO8601格式）
);

-- 记忆表（压缩 支持多种记忆类型和检索）
CREATE TABLE MemoryCompressed (
    id TEXT PRIMARY KEY,                      -- 记忆唯一标识符
    summary TEXT,                             -- 交互上下文信息[梗概内容]
    content_text_compressed TEXT,             -- 记忆内容（压缩内容）
    milvus_id TEXT,                           -- 记忆向量ID
    embedding_dimensions TEXT,                -- 对话整体向量嵌入
    importance_score REAL,                    -- 记忆重要性评分（0-1）
    created_at TEXT NOT NULL                  -- 记忆创建时间（ISO8601格式）
);

-- 用户记忆关联表
CREATE TABLE UserMemory (
    user_id TEXT,                             -- 用户ID
    memory_id TEXT                            -- 记忆ID
);

-- 句子向量表（记录句子向量和上下文增强向量）
CREATE TABLE SentenceEmbedding (
    id TEXT PRIMARY KEY,                      -- 唯一标识符
    sentence_index INTEGER,                   -- 句子在对话中的位置
    memory_id TEXT,                           -- 记忆ID
    sentence_text TEXT,                       -- 句子文本
    milvus_id TEXT,                           -- 句子向量ID
    embedding_dimensions TEXT                 -- 句子向量
);

-- 记忆与参与者关系表（记录记忆参与者信息）
CREATE TABLE MemoryParticipant (
    memory_id TEXT,                           -- 关联记忆ID
    participant_id TEXT                       -- 参与者ID
);

-- 句子与对话关系表（记录句子与记忆的关系）
CREATE TABLE SentenceMemory (
    memory_id TEXT,                           -- 关联记忆ID
    sentence_id TEXT                          -- 句子/记忆ID
);

-- 情感标记表（标记记忆和交互中的情感节点，追踪情感触发因素和强度）
CREATE TABLE EmotionalMark (
    id TEXT PRIMARY KEY,                      -- 情感标记唯一标识符
    memory_id TEXT,                           -- 关联记忆ID
    emotion_type TEXT NOT NULL,               -- 情感类型（如喜悦、悲伤、愤怒等）
    intensity REAL NOT NULL,                  -- 情感强度（0-1）
    context TEXT,                             -- 情感上下文[提炼记忆关键词]
    trigger TEXT,                             -- 情感触发因素
    created_at TEXT NOT NULL                  -- 创建时间（ISO8601格式）
);

-- 创建索引
CREATE INDEX idx_memory_importance ON Memory(importance_score);
CREATE INDEX idx_interaction_timestamp ON Interaction(created_at);
CREATE INDEX idx_memory_created ON Memory(created_at);

-- 用于全文搜索的虚拟表
-- 注意：SQLite需要启用FTS5扩展
CREATE VIRTUAL TABLE IF NOT EXISTS MemoryFTS USING FTS5(
    content_text,
    content=Memory,
    content_rowid=id
);