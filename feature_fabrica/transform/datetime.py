import numpy as np
from beartype import beartype

from feature_fabrica.transform.base import Transformation
from feature_fabrica.transform.utils import (DateTimeArray, NumericArray,
                                             StrArray, TimeDeltaArray,
                                             is_numpy_datetime_format)

DAYS_OF_WEEK = np.array(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'])

class DateTimeDifference(Transformation):
    _name_ = "datetime_diff"
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

class DateTimeArithmeticBase(Transformation):
    @beartype
    def __init__(self, time_delta: int, compute_unit: str, feature: str | None = None):
        """For simple datetime64 and timedelta64 arithmetics."""
        super().__init__()
        if compute_unit not in ['as', 'fs', 'ps', 'ns', 'us', 'ms', 's', 'm', 'h', 'D', 'W', 'M', 'Y']:
            raise ValueError(f"compute_unit= {compute_unit} is not a valid code!")

        self.time_delta = np.timedelta64(time_delta, compute_unit)
        self.feature = feature
        if self.feature:
            self.execute = self.default # type: ignore[method-assign]
        else:
            self.execute = self.with_data # type: ignore[method-assign]
    @beartype
    def default(self) -> DateTimeArray:
        raise NotImplementedError()

    @beartype
    def with_data(self, data: DateTimeArray) -> DateTimeArray:
        raise NotImplementedError()

class DateTimeAdd(DateTimeArithmeticBase):
    _name_ = "datetime_add"
    @beartype
    def with_data(self, data: DateTimeArray) -> DateTimeArray:
        return data + self.time_delta
    @beartype
    def default(self) -> DateTimeArray:
        return self.feature + self.time_delta

class DateTimeSubtract(DateTimeArithmeticBase):
    _name_ = "datetime_sub"
    @beartype
    def with_data(self, data: DateTimeArray) -> DateTimeArray:
        return data - self.time_delta
    @beartype
    def default(self) -> DateTimeArray:
        return self.feature - self.time_delta

class DateTimeExtract(Transformation):
    _name_ = "datetime_extract"
    @beartype
    def __init__(self, component: str):
        """Extracts a specific component from each datetime in the input array.

        Parameters
        ----------
        component : str
            The component to extract. Valid values are 'Y', 'M', 'D',
            'h', 'm', 's'.

        Raises
        ------
        ValueError
            If `component` is not a valid component.
        """
        super().__init__()
        valid_components = {'Y', 'M', 'D', 'h', 'm', 's'}
        if component not in valid_components:
            raise ValueError(f"Invalid component '{component}'. Valid values are {valid_components}.")
        self.component = component

    @beartype
    def execute(self, data: DateTimeArray) -> NumericArray:  # type: ignore[return]
        data_converted_object = data.astype(f'datetime64[{self.component}]').astype(object)
        # Extract component based on the specified component
        if self.component == 'Y':
            return np.array([d.year for d in data_converted_object], dtype=np.int32)
        elif self.component == 'M':
            return np.array([d.month for d in data_converted_object], dtype=np.int32)
        elif self.component == 'D':
            return  np.array([d.day for d in data_converted_object], dtype=np.int32)
        elif self.component == 'h':
            return np.array([d.hour for d in data_converted_object], dtype=np.int32)
        elif self.component == 'm':
            return np.array([d.minute for d in data_converted_object], dtype=np.int32)
        elif self.component == 's':
            return np.array([d.second for d in data_converted_object], dtype=np.int32)

class ExtractDayofWeek(Transformation):
    _name_ = "datetime_day_of_week"
    @beartype
    def __init__(self, feature: str | None = None, return_name: bool = False):
        """Extract day of the week.

        Parameters
        ----------
        feature : str | None, optional
            If set, the information will be extracted from that feature. The default is None.
        return_name : bool, optional
            If True, return the name of the day. The default is False.

        Returns
        -------
        None.
        """
        super().__init__()
        self.feature = feature
        self.return_name = return_name
        if self.feature:
            self.execute = self.default # type: ignore[method-assign]
        else:
            self.execute = self.with_data # type: ignore[method-assign]
    @beartype
    def default(self) -> NumericArray | StrArray:
        if self.feature and self.feature.dtype.type is not np.datetime64: # type: ignore[attr-defined]
            self.feature = self.feature.astype(np.datetime64) # type: ignore[attr-defined]
        result = (self.feature.astype('datetime64[D]').view('int64') - 4) % 7 # type: ignore[union-attr]
        if self.return_name:
            result = np.take(DAYS_OF_WEEK, result, axis=-1)
        return result
    @beartype
    def with_data(self, data: DateTimeArray | list[DateTimeArray]) -> NumericArray | StrArray | list[NumericArray] | list[StrArray]:
        if isinstance(data, np.ndarray):
            result = (data.astype('datetime64[D]').view('int64') - 4) % 7
            if self.return_name:
                result = np.take(DAYS_OF_WEEK, result, axis=-1)
            return result
        else:
            result_list = [(d.astype('datetime64[D]').view('int64') - 4) % 7 for d in data]
            if self.return_name:
                result_list = [np.take(DAYS_OF_WEEK, array, axis=-1) for array in result_list]
            return result_list
