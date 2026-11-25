from collections.abc import Container, Iterable
from typing import cast

from impuls import DBConnection, Task, TaskRuntime
from impuls.model import CalendarException, Date, Frequency, Route, Stop, StopTime, Trip
from impuls.tools import polish_calendar_exceptions

from . import model


class AddMetro(Task):
    def __init__(self) -> None:
        super().__init__()
        # Populated by add_calendars; kept on the instance so the exact same
        # precedence resolution drives trip/frequency generation.
        self.services: dict[str, model.Service] = {}
        self.start_date: Date | None = None
        self.end_date: Date | None = None
        self.holidays: Container[Date] = frozenset[Date]()

    def execute(self, r: TaskRuntime) -> None:
        with r.db.transaction():
            self.add_routes(r)
            self.add_stops(r)
            self.add_calendars(r)

            variants = self.parse_variants(r)

            # Resolved once from the shared Service precedence. Only calendars
            # operating at least one day get queryable trips; makeup calendars
            # are reduced to the declared directions' first trains.
            active_calendars = self.active_calendar_ids()
            makeup_calendars = self.makeup_calendar_ids()

            self.add_trips(r, variants.values(), active_calendars)
            self.add_stop_times(r, variants.values(), active_calendars)
            self.add_frequencies(
                r,
                variants.values(),
                active_calendars,
                makeup_calendars,
            )
            self.drop_empty_calendars(r.db)

    def add_routes(self, r: TaskRuntime) -> None:
        r.db.create_many(Route, map(model.parse_route, r.resources["metro_routes.csv"].csv()))

    def add_stops(self, r: TaskRuntime) -> None:
        r.db.create_many(Stop, map(model.parse_stop, r.resources["metro_stops.csv"].csv()))

    def add_calendars(self, r: TaskRuntime) -> None:
        # Get the start and end dates
        start, end = map(
            Date.from_ymd_str,
            cast(
                tuple[str, str],
                # fmt: off
                r.db
                    .raw_execute("SELECT MIN(date), MAX(date) FROM calendar_exceptions")
                    .one_must("no dates in calendar_exceptions"),
                # fmt: on
            ),
        )

        # Get the holidays
        holidays = {
            day
            for day, desc in polish_calendar_exceptions.load_exceptions(
                r.resources["calendar_exceptions.csv"],
                polish_calendar_exceptions.PolishRegion.MAZOWIECKIE,
            ).items()
            if polish_calendar_exceptions.CalendarExceptionType.HOLIDAY in desc.typ
        }

        self.start_date = start
        self.end_date = end
        self.holidays = holidays
        self.services = {}

        # Insert into DB
        for row in r.resources["metro_services.csv"].csv():
            service = model.Service.parse(row)
            self.services[service.calendar_id] = service
            r.db.create(service.as_calendar())
            r.db.create_many(
                CalendarException,
                service.as_calendar_exceptions(start, end, holidays),
            )

        return None

    def active_calendar_ids(self) -> set[str]:
        """Calendar ids operating at least one day in the feed range.

        Uses the same :meth:`model.Service.operates_on` precedence as the
        generated calendar exceptions.
        """
        if self.start_date is None or self.end_date is None:
            return set(self.services)
        return {
            calendar_id
            for calendar_id, service in self.services.items()
            if service.operates_any(self.start_date, self.end_date, self.holidays)
        }

    def operating_calendar_ids(self) -> set[str]:
        """Alias for :meth:`active_calendar_ids`."""
        return self.active_calendar_ids()

    def makeup_calendar_ids(self) -> set[str]:
        """Calendar ids representing explicit makeup (``+``) service days."""
        return {
            calendar_id
            for calendar_id, service in self.services.items()
            if service.is_makeup_service
        }

    def parse_variants(self, r: TaskRuntime) -> dict[str, model.Variant]:
        variants = dict[str, model.Variant]()

        for row in r.resources["metro_variants.csv"].csv():
            i = model.Variant.parse(row)
            variants[i.variant_id] = i

        for row in r.resources["metro_variant_stops.csv"].csv():
            i = model.VariantStop.parse(row)
            variants[i.variant_id].stops.append(i)

        for row in r.resources["metro_schedules.csv"].csv():
            i = model.Schedule.parse(row)
            variants[i.variant_id].schedules.append(i)

        return variants

    def add_trips(
        self,
        r: TaskRuntime,
        variants: Iterable[model.Variant],
        only_calendars: Container[str] | None = None,
    ) -> None:
        if only_calendars is None and self.services:
            only_calendars = self.active_calendar_ids()
        r.db.create_many(
            Trip,
            (i for v in variants for i in v.as_trips(only_calendars=only_calendars)),
        )

    def add_stop_times(
        self,
        r: TaskRuntime,
        variants: Iterable[model.Variant],
        only_calendars: Container[str] | None = None,
    ) -> None:
        if only_calendars is None and self.services:
            only_calendars = self.active_calendar_ids()
        r.db.create_many(
            StopTime,
            (i for v in variants for i in v.as_stop_times(only_calendars=only_calendars)),
        )

    def add_frequencies(
        self,
        r: TaskRuntime,
        variants: Iterable[model.Variant],
        only_calendars: Container[str] | None = None,
        first_train_only_calendars: Container[str] | None = None,
    ) -> None:
        # When invoked with the original shorter signature after
        # add_calendars(), reuse the resolved precedence instead of emitting
        # full profiles for makeup calendars.
        if first_train_only_calendars is None and self.services:
            first_train_only_calendars = self.makeup_calendar_ids()
        if only_calendars is None and self.services:
            only_calendars = self.active_calendar_ids()
        r.db.create_many(
            Frequency,
            (
                i
                for v in variants
                for i in v.as_frequencies(
                    only_calendars=only_calendars,
                    first_train_only_calendars=first_train_only_calendars,
                )
            ),
        )

    def drop_empty_calendars(self, db: DBConnection) -> None:
        db.raw_execute(
            "DELETE FROM calendars WHERE NOT EXISTS (SELECT 1 FROM calendar_exceptions "
            "WHERE calendars.calendar_id = calendar_exceptions.calendar_id)"
        )
