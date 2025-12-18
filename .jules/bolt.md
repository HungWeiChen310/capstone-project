## 2024-05-23 - [Missing Indexes on High-Volume Queries]
**Learning:** The `conversations` table is queried frequently by `sender_id` (in `get_conversation_history`) and `get_recent_conversations` (group by `sender_id`, order by `timestamp`), but lacks indexes on these columns. This leads to full table scans on every chat message and when listing recent conversations.
**Action:** Add an index on `conversations(sender_id, timestamp)` to optimize both retrieval by user and sorting by time. Also, `user_equipment_subscriptions` lacks an index on `equipment_id` for `get_subscribed_users`.
