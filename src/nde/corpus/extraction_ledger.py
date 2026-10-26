"""Nested story/cohort reservations without counting shared daily spend twice."""

from typing import override

from nde.corpus.extraction_records import ExtractionOptions, ExtractionSession
from nde.llm.ledger import CostLedger, Reservation


class ExtractionLedger(CostLedger):
    def __init__(self, session: ExtractionSession, options: ExtractionOptions) -> None:
        budget = session.config.budget
        cap = options.run_budget_usd if options.cohort_run_id else budget.per_run_usd
        super().__init__(
            options.ledger,
            per_run_usd=cap,
            per_day_usd=budget.per_day_usd,
            run_id=options.cohort_run_id or session.id,
        )
        self.story = CostLedger(
            options.ledger.with_name(options.ledger.stem + ".story-scopes.sqlite"),
            per_run_usd=budget.per_run_usd,
            run_id=options.story_run_id or session.id,
        )
        self._story_reservations: dict[int, Reservation] = {}
        self.ensure_run_spent(session.accounting_run_spent_usd)
        self.story.ensure_run_spent(session.accounting_story_spent_usd)

    @override
    def reserve(self, estimate: float) -> Reservation:
        cohort = super().reserve(estimate)
        try:
            self._story_reservations[id(cohort)] = self.story.reserve(estimate)
        except BaseException:
            super().settle(cohort, 0)
            raise
        return cohort

    @override
    def settle(self, reservation: Reservation, actual: float) -> None:
        super().settle(reservation, actual)
        self.story.settle(self._story_reservations.pop(id(reservation)), actual)

    @override
    def close(self) -> None:
        self.story.close()
        super().close()
