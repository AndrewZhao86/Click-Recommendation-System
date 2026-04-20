from dataclasses import dataclass


@dataclass(frozen=True)
class TopicConfig:
    name: str
    partitions: int
    key: str | None


USER_CLICKS = TopicConfig("user.clicks", 6, "user_id")
USER_IMPRESSIONS = TopicConfig("user.impressions", 6, "user_id")
USER_SEARCHES = TopicConfig("user.searches", 3, "user_id")
USER_PROFILE_UPDATES = TopicConfig("user.profile.updates", 6, "user_id")
USER_CLICKS_DLQ = TopicConfig("user.clicks.dlq", 1, None)

ALL_TOPICS = [
    USER_CLICKS,
    USER_IMPRESSIONS,
    USER_SEARCHES,
    USER_PROFILE_UPDATES,
    USER_CLICKS_DLQ,
]
