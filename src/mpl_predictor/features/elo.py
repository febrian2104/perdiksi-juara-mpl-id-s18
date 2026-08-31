from dataclasses import dataclass, field


@dataclass
class EloTracker:
    """Minimal two-team Elo implementation with season carryover."""

    initial_rating: float = 1500.0
    k_factor: float = 24.0
    scale: float = 400.0
    season_carryover: float = 0.75
    ratings: dict[str, float] = field(default_factory=dict)

    def rating(self, entity_id: str) -> float:
        return self.ratings.setdefault(entity_id, self.initial_rating)

    def regress_for_new_season(self, active_entities: list[str]) -> None:
        for entity_id in active_entities:
            current = self.rating(entity_id)
            self.ratings[entity_id] = self.initial_rating + self.season_carryover * (
                current - self.initial_rating
            )

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / self.scale))

    def update(self, entity_a: str, entity_b: str, actual_a: float) -> tuple[float, float]:
        rating_a = self.rating(entity_a)
        rating_b = self.rating(entity_b)
        expected_a = self.expected_score(rating_a, rating_b)
        change = self.k_factor * (actual_a - expected_a)
        self.ratings[entity_a] = rating_a + change
        self.ratings[entity_b] = rating_b - change
        return rating_a, rating_b
