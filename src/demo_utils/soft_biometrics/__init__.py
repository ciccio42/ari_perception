from enum import Enum
from messages.msg import SoftBioInfo
from collections import defaultdict, deque
from typing import List
import numpy as np
from copy import deepcopy

class BufferEmptyError(Exception): pass

class Biometrics(Enum):

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

class AGE:
    def __init__(self, age: float) -> None:
        self._age = age
    
    @property
    def age(self):
        return self._age

class GENDER(Biometrics):
    FEMALE = "F"
    MALE = "M"

class ETHNICITY(Biometrics):
    AFRICAN_AMERICAN = "African American"
    EAST_ASIAN = "East Asian"
    CAUSASIAN_LATIN = "Caucasian Latin"
    ASIAN_INDIAN = "Asian Indian"

class EMOTION(Biometrics):
    SURPRISE = "surprise"
    FEAR = "fear"
    DISGUST = "disgust"
    HAPPY = "happiness"
    SAD = "sadness"
    ANGER = "anger"
    NEUTRAL = "neutral"

class SoftBiometricsInfo:
    def __init__(self, 
                gender: GENDER = None,
                age: float = None,
                emo: EMOTION = None,
                eth: ETHNICITY = None):
        self._gender = gender
        self._age = age
        self._emo = emo
        self._eth = eth

    def to_msg(self) -> SoftBioInfo:
        msg = SoftBioInfo(age=float(self.age),
                          gender=self.gender.value,
                          ethnicity=self.ethnicity.value,
                          emotion=self.emotion.value)
        return msg
    
    @classmethod
    def from_msg(cls, msg: SoftBioInfo):
        return cls.__call__(
                     gender=GENDER(msg.gender),
                     age=msg.age,
                     emo=EMOTION(msg.emotion),
                     eth=ETHNICITY(msg.ethnicity))

    @property
    def gender(self):
        return self._gender

    @gender.setter
    def gender(self, x):
        self._gender = x
    
    @property
    def age(self):
        return self._age
    
    @age.setter
    def age(self, x):
        self._age = x
    
    @property
    def emotion(self):
        return self._emo
    
    @emotion.setter
    def emotion(self, x):
        self._emo = x
    
    @property
    def ethnicity(self):
        return self._eth

    @ethnicity.setter
    def ethnicity(self, x):
        self._eth = x
    
    def __str__(self):
        return f"G: {self.gender} - A: {self.age}, E: {self.emotion}, ETH: {self.ethnicity}"

class Collector:

    AGE_LIMIT = 17

    def __init__(self):
        maxlen_age = 60
        self._maxlen_others = 7
        self._buffer_dict = defaultdict(lambda: deque(maxlen=maxlen_age))
    
    def add(self, track_id, info: SoftBiometricsInfo):
        self._buffer_dict[track_id].append(info)
    
    def get(self, track_id) -> SoftBiometricsInfo:
        age = self._get(track_id, "age")
        emotion = self._get(track_id, "emotion")
        gender = self._get(track_id, "gender")
        ethnicity = self._get(track_id, "ethnicity")
        info_res = SoftBiometricsInfo(age=age,
                                        gender=gender,
                                        emo=emotion,
                                        eth=ethnicity)
        return info_res

    def _get(self, track_id, attribute):
        data_list = []
        info_list = self._buffer_dict[track_id]
        for info in info_list:
            info: SoftBiometricsInfo
            data_list.append(getattr(info, attribute))
        
        if len(data_list) == 0:
            raise BufferEmptyError

        if attribute == "age":
            value = np.max(data_list).item()
            if value < Collector.AGE_LIMIT:
                value = Collector.AGE_LIMIT
                
        else:
            data_list = data_list[:-self._maxlen_others:-1]
            unique, counts = np.unique(data_list, return_counts=True)
            idx = np.argmax(counts)
            value = unique[idx]
        return value