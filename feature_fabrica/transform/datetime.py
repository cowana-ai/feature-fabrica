import numpy as np
from beartype import beartype

from feature_fabrica.transform.base import Transformation
from feature_fabrica.transform.utils import (DateTimeArray, StrArray,
                                             TimeDeltaArray,
                                             is_numpy_datetime_format)


class DateTimeDifference(Transformation):
    @beartype
    def __init__(self, initial_datetime: str | None = None, end_datetime: str | None = None, compute_unit: str | None = None):
        """This transformation computes the difference between a provided datetime array and an initial or end datetime.
        The difference is computed in the specified `compute_unit` (e.g., seconds, minutes, hours). This can be used to
        compute such features as recency or time difference.

        Parameters
        ----------
        initial_datetime : str | None, optional (can be feature)
            The initial datetime string, formatted as a NumPy-compatible datetime.
            If set, the difference will be computed as the data minus the initial
            datetime. Either `initial_datetime` or `end_datetime` must be set, but
            not both.
        end_datetime : str | None, optional (can be feature)
            The end datetime string, formatted as a NumPy-compatible datetime.
            If set, the difference will be computed as the end datetime minus the
            data. Either `end_datetime` or `initial_datetime` must be set, but
            not both.
        compute_unit : str | None, optional
            The unit in which to compute the datetime difference, such as 's' for
            seconds or 'D' for days. Valid values are ['as', 'fs', 'ps', 'ns', 'us',
            'ms', 's', 'm', 'h', 'D', 'W', 'M', 'Y']. The default is None, which
            uses the default datetime unit.

        Raises
        ------
        ValueError
            If both `initial_datetime` and `end_datetime` are set or if neither is
            set. Also raised if `compute_unit` is not a valid code.
        """
        super().__init__()
        if (initial_datetime and end_datetime) or (initial_datetime is None and end_datetime is None):
            raise ValueError("Only one of 'initial_datetime' or 'end_datetime' should be set!")
        if compute_unit and compute_unit not in ['as', 'fs', 'ps', 'ns', 'us', 'ms', 's', 'm', 'h', 'D', 'W', 'M', 'Y']:
            raise ValueError(f"compute_unit= {compute_unit} is not a valid code!")

        self.initial_datetime = initial_datetime
        self.end_datetime = end_datetime

        if self.initial_datetime and is_numpy_datetime_format(self.initial_datetime):
            self.initial_datetime = np.array(self.initial_datetime, dtype=np.datetime64)
        elif self.end_datetime and is_numpy_datetime_format(self.end_datetime):
            self.end_datetime = np.array(self.end_datetime, dtype=np.datetime64)

        self.compute_unit = compute_unit

    @beartype
    def execute(self, data: StrArray | DateTimeArray) ->  TimeDeltaArray: # type: ignore
        # If it's a full datetime, compare date and time units
        if data.dtype.type is not np.datetime64: # type: ignore
            data = data.astype(np.datetime64) # type: ignore

        if self.initial_datetime and self.initial_datetime.dtype.type is not np.datetime64: # type: ignore[attr-defined]
            self.initial_datetime = self.initial_datetime.astype(np.datetime64) # type: ignore[attr-defined]

        if self.end_datetime and self.end_datetime.dtype.type is not np.datetime64: # type: ignore[attr-defined]
            self.end_datetime = self.end_datetime.astype(np.datetime64) # type: ignore[attr-defined]


        result = data - self.initial_datetime if self.initial_datetime else self.end_datetime - data # type: ignore
        if self.compute_unit:
            result = result.astype(f'timedelta64[{self.compute_unit}]')

        return result
