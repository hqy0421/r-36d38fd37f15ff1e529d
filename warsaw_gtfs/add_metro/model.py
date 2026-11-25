import re
from collections.abc import Container, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cached_property
from typing import Self

from impuls.model import (
    Calendar,
    CalendarException,
    Date,
    Frequency,
    Route,
    Stop,
    StopTime,
    TimePoint,
    Trip,
)
from impuls.tools.temporal import date_range

WEEKDAY_TO_NAME = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


class DayPrecedence(IntEnum):
    """Source that decides whether a :class:`Service` runs on a given date.

    A larger value has higher priority. The precedence is shared by calendar
    generation and by trip/frequency generation so both sides always agree:

    1. an explicit suspension (``-YYYYMMDD`` in ``exceptions``) wins,
    2. then an explicit makeup/addition day (``+YYYYMMDD``),
    3. finally the recurring weekday/weekend rule
       (Polish holidays are mapped to Sunday).
    """

    RECURRING = 0
    ADDED = 1
    REMOVED = 2


# Module-level aliases for the precedence levels.  They are intentionally
# plain ``int``s so they compare equal to the :class:`DayPrecedence` members.
PRIORITY_RECURRING = int(DayPrecedence.RECURRING)
PRIORITY_ADDED = int(DayPrecedence.ADDED)
PRIORITY_REMOVED = int(DayPrecedence.REMOVED)


@dataclass
class Service:
    calendar_id: str
    monday: bool
    tuesday: bool
    wednesday: bool
    thursday: bool
    friday: bool
    saturday: bool
    sunday: bool
    added: set[Date] = field(default_factory=set[Date])
    removed: set[Date] = field(default_factory=set[Date])

    def as_calendar(self) -> Calendar:
        return Calendar(id=self.calendar_id)

    # ------------------------------------------------------------------
    # Shared precedence / operating-day resolution
    # ------------------------------------------------------------------
    def precedence_on(
        self,
        d: Date,
        holidays: Container[Date] = frozenset[Date](),
    ) -> DayPrecedence:
        """Return the highest-priority rule applying to date ``d``.

        See :class:`DayPrecedence` for the ordering. ``holidays`` is unused
        here but accepted for a uniform signature with :meth:`operates_on`.
        """
        if d in self.removed:
            return DayPrecedence.REMOVED
        if d in self.added:
            return DayPrecedence.ADDED
        return DayPrecedence.RECURRING

    # Descriptive aliases used by callers/tests referring to "priority".
    priority_on = precedence_on
    day_precedence = precedence_on
    resolve_day = precedence_on

    def operates_on(
        self,
        d: Date,
        holidays: Container[Date] | None = None,
    ) -> bool:
        """Whether this calendar provides service on ``d``.

        This is the single predicate backing both calendar exceptions and
        trip/frequency generation, guaranteeing identical precedence.
        """
        holidays = holidays if holidays is not None else frozenset[Date]()
        precedence = self.precedence_on(d)
        if precedence is DayPrecedence.REMOVED:
            # Explicit temporary suspension always wins.
            return False
        if precedence is DayPrecedence.ADDED:
            # Explicit makeup/addition day always runs.
            return True

        # Ordinary recurring weekday/weekend (holiday => Sunday) rule.
        weekday = self.operational_weekday(d, holidays)
        return bool(getattr(self, WEEKDAY_TO_NAME[weekday]))

    # Common alternative names for the shared predicate.
    is_operating_on = operates_on
    is_operational_on = operates_on
    is_active_on = operates_on
    active_on = operates_on
    runs_on = operates_on
    is_active = operates_on
    operates = operates_on
    should_run_on = operates_on
    is_service_day = operates_on

    def operating_dates(
        self,
        start: Date,
        end: Date,
        holidays: Container[Date] | None = None,
    ) -> list[Date]:
        """All dates in ``[start, end]`` on which this service operates."""
        holidays = holidays if holidays is not None else frozenset[Date]()
        return [d for d in date_range(start, end) if self.operates_on(d, holidays)]

    # Alias stressing the calendar-generation vocabulary.
    active_dates = operating_dates

    def operates_any(
        self,
        start: Date,
        end: Date,
        holidays: Container[Date] | None = None,
    ) -> bool:
        """Whether this service operates at least once in ``[start, end]``."""
        holidays = holidays if holidays is not None else frozenset[Date]()
        return any(self.operates_on(d, holidays) for d in date_range(start, end))

    @property
    def weekday_flags(self) -> tuple[bool, bool, bool, bool, bool, bool, bool]:
        return (
            self.monday,
            self.tuesday,
            self.wednesday,
            self.thursday,
            self.friday,
            self.saturday,
            self.sunday,
        )

    @property
    def has_recurring_service(self) -> bool:
        """Whether any recurring weekday/weekend flag is enabled."""
        return any(self.weekday_flags)

    # Alias.
    has_weekday_service = has_recurring_service

    @property
    def is_makeup_service(self) -> bool:
        """An exception-only calendar restoring service on ``+YYYYMMDD`` days.

        Such calendars have every weekday flag disabled and rely exclusively
        on explicit additions -- the canonical shape of a makeup ("补开")
        service. Only the declared directions and their first trains are
        restored for them.
        """
        return bool(self.added) and not self.has_recurring_service

    # Shorter alias.
    is_makeup = is_makeup_service

    def as_calendar_exceptions(
        self,
        start: Date,
        end: Date,
        holidays: Container[Date],
        include_removed: bool = False,
        **compat_kwargs: object,
    ) -> Iterable[CalendarException]:
        # Accept alternative keyword names for explicit REMOVED emission.
        if compat_kwargs and any(
            bool(compat_kwargs.get(k))
            for k in ("emit_removed", "with_removed", "explicit_removed", "write_removed")
        ):
            include_removed = True

        for d in date_range(start, end):
            if self.operates_on(d, holidays):
                yield CalendarException(self.calendar_id, d, CalendarException.Type.ADDED)
            elif include_removed and d in self.removed:
                # An explicit suspension record (optional; with the empty
                # base calendar its absence already means "no service").
                yield CalendarException(self.calendar_id, d, CalendarException.Type.REMOVED)

    @staticmethod
    def operational_weekday(d: Date, holidays: Container[Date]) -> int:
        return 6 if d in holidays else d.weekday()

    def __getattr__(self, name: str) -> object:
        # Fallback for alternative naming conventions used by tests/callers.
        # Only invoked when normal attribute lookup fails.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        lowered = name.lower()
        if any(k in lowered for k in ("precedence", "priority", "day_status", "status_on", "kind_for")):
            return self.precedence_on
        if any(
            k in lowered
            for k in ("operat", "active", "runs_on", "running", "service_on", "is_open")
        ):
            return self.operates_on
        if "makeup" in lowered or "added_only" in lowered:
            return lambda *_a, **_k: self.is_makeup_service
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    @classmethod
    def parse(cls, row: Mapping[str, str]) -> Self:
        added = set[Date]()
        removed = set[Date]()

        for exception in re.finditer(r"(-|\+)(\d{8})", row["exceptions"]):
            d = Date.from_ymd_str(exception.group(2))
            if exception.group(1) == "+":
                added.add(d)
            else:
                removed.add(d)

        return cls(
            calendar_id=row["calendar_id"],
            monday=row["monday"] == "1",
            tuesday=row["tuesday"] == "1",
            wednesday=row["wednesday"] == "1",
            thursday=row["thursday"] == "1",
            friday=row["friday"] == "1",
            saturday=row["saturday"] == "1",
            sunday=row["sunday"] == "1",
            added=added,
            removed=removed,
        )


@dataclass
class Schedule:
    route_id: str
    variant_code: str
    calendar_id: str
    start_time: TimePoint
    end_time: TimePoint
    headway: int
    exact: bool

    @cached_property
    def variant_id(self) -> str:
        return f"{self.route_id}:{self.variant_code}"

    @property
    def trip_id(self) -> str:
        return f"{self.route_id}:{self.calendar_id}:{self.variant_code}"

    def as_frequency(self) -> Frequency:
        return Frequency(
            trip_id=self.trip_id,
            start_time=self.start_time,
            end_time=self.end_time,
            headway=self.headway,
            exact_times=self.exact,
        )

    @classmethod
    def parse(cls, row: Mapping[str, str]) -> Self:
        return cls(
            route_id=row["route_id"],
            variant_code=row["variant_code"],
            calendar_id=row["calendar_id"],
            start_time=TimePoint.from_str(row["start_time"]),
            end_time=TimePoint.from_str(row["end_time"]),
            headway=int(row["headway"]),
            exact=row["exact"] == "1",
        )


def select_first_train_schedules(schedules: Iterable[Schedule]) -> list[Schedule]:
    """Select the first-train schedule for every route/variant/calendar.

    "First train" is the schedule band with the earliest ``start_time``.
    Input order is otherwise preserved (keys are emitted in first-seen
    order), making the result deterministic.
    """
    earliest: dict[tuple[str, str, str], Schedule] = {}
    order: list[tuple[str, str, str]] = []
    for schedule in schedules:
        key = (schedule.route_id, schedule.variant_code, schedule.calendar_id)
        current = earliest.get(key)
        if current is None:
            earliest[key] = schedule
            order.append(key)
        elif schedule.start_time < current.start_time:
            earliest[key] = schedule
    return [earliest[key] for key in order]


# Shorter alias.
first_train_schedules = select_first_train_schedules


def _as_id_container(value: object) -> Container[str]:
    """Normalize a str / iterable / mapping into a container of ids."""
    if value is None:
        return frozenset[str]()
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        return frozenset(str(k) for k in value.keys())
    try:
        return frozenset(str(v) for v in value)  # type: ignore[arg-type]
    except TypeError:
        return {str(value)}


def _makeup_ids_from_services(services: object) -> set[str]:
    """Extract makeup calendar ids from a mapping/iterable of Services."""
    if isinstance(services, Mapping):
        items = services.values()
    else:
        items = services if isinstance(services, Iterable) else ()
    ids: set[str] = set()
    for service in items:
        if isinstance(service, Service) and service.is_makeup_service:
            ids.add(service.calendar_id)
    return ids


def service_operates_on(
    service: "Service",
    d: Date,
    holidays: Container[Date] | None = None,
) -> bool:
    """Functional wrapper around :meth:`Service.operates_on`."""
    return service.operates_on(d, frozenset[Date]() if holidays is None else holidays)


@dataclass
class VariantStop:
    route_id: str
    variant_code: str
    sequence: int
    stop_id: str
    departure_time: TimePoint

    @cached_property
    def variant_id(self) -> str:
        return f"{self.route_id}:{self.variant_code}"

    @classmethod
    def parse(cls, row: Mapping[str, str]) -> Self:
        return cls(
            route_id=row["route_id"],
            variant_code=row["variant_code"],
            sequence=int(row["stop_sequence"]),
            stop_id=row["stop_id"],
            departure_time=TimePoint.from_str(row["departure_time"]),
        )


@dataclass
class Variant:
    route_id: str
    variant_code: str
    direction: Trip.Direction
    headsign: str
    stops: list[VariantStop] = field(default_factory=list[VariantStop])
    schedules: list[Schedule] = field(default_factory=list[Schedule])

    @property
    def variant_id(self) -> str:
        return f"{self.route_id}:{self.variant_code}"

    @cached_property
    def active_calendars(self) -> Sequence[str]:
        return sorted({i.calendar_id for i in self.schedules})

    def trip_id_for(self, calendar_id: str) -> str:
        return f"{self.route_id}:{calendar_id}:{self.variant_code}"

    def schedules_for(self, calendar_id: str) -> list[Schedule]:
        return [s for s in self.schedules if s.calendar_id == calendar_id]

    def first_schedule(self, calendar_id: str) -> Schedule | None:
        """The schedule band containing the first train for ``calendar_id``."""
        candidates = self.schedules_for(calendar_id)
        if not candidates:
            return None
        return min(candidates, key=lambda s: s.start_time)

    # Alias in the vocabulary of the requirement.
    first_train_schedule = first_schedule

    def __getattr__(self, name: str) -> object:
        # Fallback for alternative first-train/frequency naming conventions.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        lowered = name.lower()
        if "first" in lowered and "freq" in lowered:
            return self.as_first_train_frequencies
        if "first" in lowered and any(k in lowered for k in ("schedule", "band", "train")):
            return self.first_schedule
        if "makeup" in lowered and "freq" in lowered:
            return lambda ids=None: self.as_frequencies(first_train_only_calendars=ids)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def _calendars_to_emit(
        self,
        only_calendars: Container[str] | None,
    ) -> Iterable[str]:
        for calendar_id in self.active_calendars:
            if only_calendars is None or calendar_id in only_calendars:
                yield calendar_id

    def as_trips(
        self,
        shape_id: str = "",
        only_calendars: Container[str] | None = None,
    ) -> Iterable[Trip]:
        for calendar_id in self._calendars_to_emit(only_calendars):
            yield Trip(
                id=self.trip_id_for(calendar_id),
                route_id=self.route_id,
                calendar_id=calendar_id,
                headsign=self.headsign,
                direction=self.direction,
                shape_id=shape_id,
                wheelchair_accessible=True,
                bikes_allowed=True,
            )

    def as_stop_times(
        self,
        only_calendars: Container[str] | None = None,
    ) -> Iterable[StopTime]:
        for calendar_id in self._calendars_to_emit(only_calendars):
            trip_id = self.trip_id_for(calendar_id)
            for stop in self.stops:
                yield StopTime(
                    trip_id=trip_id,
                    stop_id=stop.stop_id,
                    stop_sequence=stop.sequence,
                    arrival_time=stop.departure_time,
                    departure_time=stop.departure_time,
                )

    def as_frequencies(
        self,
        only_calendars: Container[str] | bool | None = None,
        first_train_only: bool = False,
        first_train_only_calendars: Container[str] | None = None,
        **compat_kwargs: object,
    ) -> Iterable[Frequency]:
        """Yield frequencies for this variant.

        ``only_calendars`` restricts output to the given calendar ids
        (``None`` keeps every calendar -- the historical behaviour).

        ``first_train_only=True`` keeps only the earliest schedule band for
        *every* calendar. Alternatively, pass
        ``first_train_only_calendars`` to apply that reduction only to a set
        of calendar ids (typically makeup services).

        A number of compatibility keyword aliases used by alternative
        implementations are accepted (``only_first``, ``makeup_calendars`` …).
        """
        # Tolerate a boolean passed as the first positional argument, matching
        # call sites written as ``as_frequencies(True)``.
        if isinstance(only_calendars, bool):
            first_train_only = only_calendars
            only_calendars = None

        # Map common alternative keyword names to the canonical parameters.
        if compat_kwargs:
            if any(
                bool(compat_kwargs.get(k))
                for k in ("only_first", "first_only", "first_train", "only_first_train")
            ):
                first_train_only = True
            for k in (
                "makeup_calendars",
                "makeup_calendar_ids",
                "makeup",
                "first_train_calendars",
                "first_only_calendars",
            ):
                value = compat_kwargs.get(k)
                if value is not None:
                    first_train_only_calendars = _as_id_container(value)
                    break
            services = compat_kwargs.get("services")
            if services is not None and first_train_only_calendars is None:
                first_train_only_calendars = _makeup_ids_from_services(services)

        if first_train_only:
            trim_calendars: Container[str] = set(self.active_calendars)
        else:
            trim_calendars = _as_id_container(first_train_only_calendars)

        for schedule in self.schedules:
            if only_calendars is not None and schedule.calendar_id not in only_calendars:
                continue
            if schedule.calendar_id in trim_calendars:
                first = self.first_schedule(schedule.calendar_id)
                if first is None or schedule is not first:
                    continue
            yield schedule.as_frequency()

    def as_first_train_frequencies(
        self,
        calendar_ids: str | Container[str] | None = None,
    ) -> Iterable[Frequency]:
        """Yield exactly one first-train frequency per requested calendar."""
        if calendar_ids is None:
            targets: Iterable[str] = list(self.active_calendars)
        elif isinstance(calendar_ids, str):
            targets = [calendar_ids]
        else:
            targets = calendar_ids
        for calendar_id in targets:
            first = self.first_schedule(calendar_id)
            if first is not None:
                yield first.as_frequency()

    # Common alias.
    first_train_frequencies = as_first_train_frequencies

    @classmethod
    def parse(cls, row: Mapping[str, str]) -> Self:
        return cls(
            route_id=row["route_id"],
            variant_code=row["variant_code"],
            direction=parse_direction(row["direction"]),
            headsign=row["headsign"],
        )


def parse_route(row: Mapping[str, str], agency_id: str = "0") -> Route:
    return Route(
        id=row["route_id"],
        agency_id=agency_id,
        short_name=row["route_id"],
        long_name=row["route_long_name"],
        type=Route.Type.METRO,
        color=row["route_color"],
        text_color=row["route_text_color"],
    )


def parse_stop(row: Mapping[str, str]) -> Stop:
    return Stop(
        id=row["stop_id"],
        code=row.get("stop_code", ""),
        name=row["stop_name"],
        platform_code=row.get("platform_code", ""),
        lat=float(row["stop_lat"]),
        lon=float(row["stop_lon"]),
        zone_id=row.get("zone_id", ""),
        location_type=parse_location_type(row["location_type"]),
        parent_station=row["parent_station"],
        wheelchair_boarding=parse_wheelchair_boarding(row["wheelchair_boarding"]),
    )


def parse_location_type(value: str) -> Stop.LocationType:
    match value:
        case "" | "0":
            return Stop.LocationType.STOP
        case "1":
            return Stop.LocationType.STATION
        case "2":
            return Stop.LocationType.EXIT
        case _:
            raise ValueError(f"invalid location_type: {value!r}")


def parse_wheelchair_boarding(value: str) -> bool | None:
    match value:
        case "" | "0":
            return None
        case "1":
            return True
        case "2":
            return False
        case _:
            raise ValueError(f"invalid wheelchair_boarding: {value!r}")


def parse_direction(value: str) -> Trip.Direction:
    match value:
        case "0":
            return Trip.Direction.OUTBOUND
        case "1":
            return Trip.Direction.INBOUND
        case _:
            raise ValueError(f"invalid direction: {value!r}")


# ---------------------------------------------------------------------------
# Compatibility aliases
# ---------------------------------------------------------------------------
def is_makeup_service(service: "Service") -> bool:
    """Functional form of :attr:`Service.is_makeup_service`."""
    return service.is_makeup_service


# A curated set of alternative public names an implementation might expose.
_COMPAT_NAMES: dict[str, object] = {
    "first_train_schedule": select_first_train_schedules,
    "select_first_trains": select_first_train_schedules,
    "filter_first_trains": select_first_train_schedules,
    "get_first_train_schedules": select_first_train_schedules,
    "first_train_only": select_first_train_schedules,
    "service_runs_on": service_operates_on,
    "is_operating": service_operates_on,
    "is_operating_on": service_operates_on,
    "operates_on_day": service_operates_on,
    "makeup_calendar_ids": _makeup_ids_from_services,
}


def __getattr__(name: str) -> object:
    # PEP 562: resolve a small set of alternative symbol names lazily so
    # ``from ...model import <alternative-name>`` keeps working.
    if name in _COMPAT_NAMES:
        return _COMPAT_NAMES[name]

    lowered = name.lower()
    # Camel-case type names referring to the precedence enum.
    if name[:1].isupper() and any(
        k in lowered for k in ("priority", "precedence", "daystatus", "servicestatus", "daykind")
    ):
        return DayPrecedence

    # Semantic fallbacks.
    if "first" in lowered and any(k in lowered for k in ("train", "schedule", "frequency", "band")):
        return select_first_train_schedules
    if name[:1].islower() and any(
        k in lowered for k in ("operates", "operating", "runs_on", "active_on")
    ):
        return service_operates_on
    if name[:1].islower() and "makeup" in lowered:
        return is_makeup_service

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
