from __future__ import annotations

from typing import List
from .base import BaseModel


sessions_class: List[type[BaseModel]] = []
sessions_names: List[str] = []

"""
BiRefNet Family
"""
from .birefnet_general import BiRefNetGeneral

sessions_class.append(BiRefNetGeneral)
sessions_names.append(BiRefNetGeneral.name())

from .birefnet_general_lite import BiRefNetGeneralLite

sessions_class.append(BiRefNetGeneralLite)
sessions_names.append(BiRefNetGeneralLite.name())

from .birefnet_portrait import BiRefNetPortrait

sessions_class.append(BiRefNetPortrait)
sessions_names.append(BiRefNetPortrait.name())

from .birefnet_dis import BiRefNetDIS

sessions_class.append(BiRefNetDIS)
sessions_names.append(BiRefNetDIS.name())

from .birefnet_hrsod import BiRefNetHRSOD

sessions_class.append(BiRefNetHRSOD)
sessions_names.append(BiRefNetHRSOD.name())

from .birefnet_cod import BiRefNetCOD

sessions_class.append(BiRefNetCOD)
sessions_names.append(BiRefNetCOD.name())

from .birefnet_massive import BiRefNetMassive

sessions_class.append(BiRefNetMassive)
sessions_names.append(BiRefNetMassive.name())

"""
ISNet Family
"""
from .dis_anime import Dis as DisAnime

sessions_class.append(DisAnime)
sessions_names.append(DisAnime.name())

from .dis_general import Dis as DisGeneral

sessions_class.append(DisGeneral)
sessions_names.append(DisGeneral.name())

"""
SAM Family
"""
from .sam import Sam

sessions_class.append(Sam)
sessions_names.append(Sam.name())

"""
U2Net Family
"""
from .silueta import Silueta

sessions_class.append(Silueta)
sessions_names.append(Silueta.name())

from .u2net_cloth import U2netCloth

sessions_class.append(U2netCloth)
sessions_names.append(U2netCloth.name())

from .u2net_custom import U2netCustom

sessions_class.append(U2netCustom)
sessions_names.append(U2netCustom.name())

from .u2net_human import U2netHuman

sessions_class.append(U2netHuman)
sessions_names.append(U2netHuman.name())

from .u2net import U2net

sessions_class.append(U2net)
sessions_names.append(U2net.name())

from .u2netp import U2netp

sessions_class.append(U2netp)
sessions_names.append(U2netp.name())
