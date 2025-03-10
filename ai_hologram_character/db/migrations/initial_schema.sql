-- 用户基础表
-- 用户多维特详细征表（通过交互记录更新）
CREATE TABLE user_profile (
    user_id INTEGER PRIMARY KEY,              -- 用户唯一标识符
    voice_print_id TEXT                       -- 用户声纹ID 初期唯一ID（但是对于对话内容抽取出来的用户是没有声纹的）
    -- face_print_id TEXT,                       -- 用户人脸ID
);

-- 用户基本信息表
CREATE TABLE external_info (
    user_id INTEGER PRIMARY KEY,              -- 用户唯一标识符
    -- 基本信息（姓名，年龄，出生，国籍，民族，籍贯）
    nickname TEXT,                            -- 用户昵称
    name TEXT,                                -- 用户姓名
    gender TEXT,                              -- 用户性别
    age INTEGER,                              -- 用户年龄
    birthday TEXT,                            -- 出生日期（YYYY-MM-DD格式）
    telephone TEXT,                           -- 联系电话
    address TEXT,                             -- 联系地址
    email TEXT,                                -- 电子邮箱
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
CREATE TABLE internal_info (
    user_id INTEGER PRIMARY KEY,              -- 用户ID
    -- 健康状况
    health_status TEXT,                       -- 健康状况
    -- 外在特征
    physical_features TEXT,                   -- 体貌特征（包含外貌特征[身高体重等]，穿着打扮等）
    expression_style TEXT,                    -- 表达方式（包含语言表达，肢体语言，面部表情等方面的习惯性行为）
    social_behavior TEXT,                     -- 社交行为（包含社交技巧，人际交往方式，社交圈子等）
    -- 内在特征
    emotional_traits TEXT,                    -- 情感特征（包含情绪稳定性，敏感度等）
    personality TEXT                          -- 自我认知，身份认同，人生目标，价值观体系等
);

-- 目前保留
-- 用户关系表（每次对话结束，推断参与者之间的关系，并对不确定的重要关系提出追问，重点是姓名）
-- CREATE TABLE Relationship (
--     id TEXT PRIMARY KEY,                      -- 关系唯一标识符
--     user_id TEXT,                             -- 用户ID
--     target_user_id TEXT,                      -- 目标用户ID
--     relationship_type TEXT,                   -- （亲人（具体称谓）/恋人/朋友/同事等） 如果有多个身份，可以"同事，朋友"这样定义
--     strength REAL,                            -- 关系强度（0-1）
--     last_updated TEXT NOT NULL                -- 最后更新时间（ISO8601格式）
-- );

-- 目前保留
-- 隐私信息表（通过交互记录更新，目前保留）
-- CREATE TABLE Secret (
--     id TEXT PRIMARY KEY,                      -- 隐私信息唯一标识符
--     user_id TEXT,                             -- 用户ID
--     secret_type TEXT NOT NULL,                -- 隐私信息类型（个人/家庭/财务等）
--     secret_content TEXT NOT NULL,             -- 隐私信息内容
--     last_updated TEXT NOT NULL                -- 最后更新时间（ISO8601格式）
-- );

-- 交互记录表
CREATE TABLE interaction (
    interaction_id INTEGER PRIMARY KEY,       -- 交互记录ID
    memory_id INTEGER,                        -- 关联记忆ID[1对1]
    user_intent TEXT,                         -- 用户意图
    context_summary TEXT,                     -- 交互上下文信息[梗概内容]
    -- 交互环境信息，记录交互发生的环境上下文，增强情境感知能力
    -- ambient_factors TEXT,                     -- 环境因素
    created_at TEXT NOT NULL                  -- 创建时间（ISO8601格式）
);

-- 角色交互记录关系表（交互记录与用户：一对多）
CREATE TABLE interaction_user (
    interaction_id INTEGER,                   -- 交互记录ID
    user_id INTEGER                           -- 用户ID
);

-- 记忆表（未压缩 支持多种记忆类型和检索）
CREATE TABLE memory (
    memory_id INTEGER PRIMARY KEY,            -- 记忆唯一标识符
    context_summary TEXT NOT NULL,            -- 交互上下文信息[梗概内容：主题，人物等]
    content_text TEXT NOT NULL,               -- 记忆内容（详细对话文本）
    is_compressed INTEGER NOT NULL,           -- 是否压缩（0-未压缩，1-压缩）
    milvus_id INTEGER NOT NULL,               -- 记忆向量ID
    embedding_dimensions TEXT NOT NULL,       -- 对话整体向量（基于对话梗概context_summary生成）
    importance_score REAL NOT NULL,           -- 记忆重要性评分（0-1）
    emotion_type TEXT NOT NULL,               -- 情感类型（如喜悦、悲伤、愤怒等）
    emotion_intensity REAL NOT NULL,          -- 情感强度（0-1）
    emotion_trigger TEXT  NOT NULL,           -- 情感触发因素
    created_at TEXT NOT NULL                  -- 记忆创建时间（ISO8601格式）
);

-- 记忆表（压缩 支持多种记忆类型和检索）
-- CREATE TABLE MemoryCompressed (
--     memory_id TEXT PRIMARY KEY,               -- 记忆唯一标识符
--     summary TEXT,                             -- 交互上下文信息[梗概内容]
--     content_text_compressed TEXT,             -- 记忆内容（压缩内容）
--     milvus_id TEXT,                           -- 记忆向量ID
--     embedding_dimensions TEXT,                -- 对话整体向量
--     importance_score REAL,                    -- 记忆重要性评分（0-1）
--     created_at TEXT NOT NULL                  -- 记忆创建时间（ISO8601格式）
-- );

-- 句子向量表（记录句子向量和上下文增强向量） 只存储有实际意义的内容<需要过滤机制>
CREATE TABLE sentence (
    sentence_id INTEGER PRIMARY KEY,             -- 句子向量唯一标识符
    memory_id INTEGER NOT NULL,                  -- 关联记忆ID
    sentence_index INTEGER NOT NULL,             -- 句子在对话中的位置（0,1,2,3....N）
    sentence_text TEXT NOT NULL,                 -- 句子文本
    milvus_id INTEGER NOT NULL,                  -- 句子向量ID（与该行句子ID一致）
    embedding_dimensions TEXT NOT NULL,          -- 句子向量
    created_at TEXT NOT NULL                      -- 句子创建时间（ISO8601格式）
);

-- 自增ID管理表
CREATE TABLE id_generator_a (
    id INTEGER PRIMARY KEY AUTOINCREMENT         -- 自增ID（interaction_id/memory_id/milvus_id[memory]）
);
CREATE TABLE id_generator_b (
    id INTEGER PRIMARY KEY AUTOINCREMENT         -- 自增ID（sentence_id/milvus_id[sentence]）
);

-- 创建索引
CREATE INDEX idx_external_info ON external_info(name);


-- 用于全文搜索的虚拟表
-- 注意：SQLite需要启用FTS5扩展
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING FTS5(
    content_text,
    content=memory,
    content_rowid=memory_id
);