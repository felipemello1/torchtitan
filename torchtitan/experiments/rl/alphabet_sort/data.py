# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic local data generation for AlphabetSort."""

from __future__ import annotations

import random
from dataclasses import dataclass


NAMES: tuple[str, ...] = (
    "EnginDurgun",
    "DenizCakir",
    "NurtenAkman",
    "MinaFarzan",
    "OmarYildiz",
    "LenaKovac",
    "ArunPatel",
    "SofiaNiemi",
    "TariqHassan",
    "ElenaRossi",
    "JonasBerg",
    "IrisLund",
    "MateoSilva",
    "NoorRahman",
    "ClaraVogel",
    "RaviMenon",
    "YukiTanaka",
    "NadiaPopov",
    "FelixWeber",
    "AishaKhan",
    "MartaNowak",
    "LeoSchmidt",
    "SaraDemir",
    "IvanPetrov",
    "LinaSato",
    "PavelNovak",
    "RosaGarcia",
    "NikoVirtanen",
    "AmirAbbasi",
    "EvaHorvat",
    "AaronBell",
    "AdamCole",
    "AdrianDean",
    "AidenEdge",
    "AlanFord",
    "AlexGray",
    "AmirHale",
    "AndreIver",
    "ArjunJude",
    "AveryKim",
    "BaoLane",
    "BrianMack",
    "BrunoNoah",
    "CalebOak",
    "CarterPace",
    "CesarQuinn",
    "CodyReed",
    "DarioSage",
    "DavidTrent",
    "DeanUmer",
    "DylanVega",
    "EliWong",
    "EmilXu",
    "EzraYork",
    "FinnZane",
    "GabeAsh",
    "HankBay",
    "IanCox",
    "IsaacDay",
    "JackEast",
    "JadenFalk",
    "JakeGold",
    "JaredHolt",
    "JasonInks",
    "JamalJett",
    "JavierKane",
    "JinLim",
    "JonahMarsh",
    "JoseNash",
    "JoshuaOrth",
    "KaiPark",
    "KenjiQuill",
    "KevinRose",
    "KhalidSky",
    "LamarTate",
    "LeonUrban",
    "LiamVail",
    "LoganWilde",
    "LucasXena",
    "LuisYale",
    "ManuelZora",
    "MarcusAtta",
    "MarkBirch",
    "MateoCrane",
    "MattDixon",
    "MauricioEve",
    "MaxFisher",
    "MiguelGreer",
    "MohammedHill",
    "NateIndus",
    "NeilJones",
    "NicoKraft",
    "NoahLowe",
    "OliverMarks",
    "OmarNorth",
    "OwenOaks",
    "PabloPine",
    "PatrickQuartz",
    "PaulRivers",
    "PedroStone",
    "PeterTucker",
    "PranavUp",
    "QuincyVale",
    "RafaelWebb",
    "RahulXi",
    "RaviYork",
    "RyanZero",
    "SamArc",
    "SamuelBole",
    "SantiagoCrisp",
    "SebastianDell",
    "SergioErin",
    "ShawnFawn",
    "SimonGlen",
    "StevenHank",
    "TariqIce",
    "TheoJule",
    "ThomasKelp",
    "TimLock",
    "TobiasMott",
    "TomasNev",
    "TravisOrr",
    "TylerPike",
    "UmarQuay",
    "VanceRae",
    "VictorSpry",
    "VihanTown",
    "VincentUlm",
    "WalidVale",
    "WalterWisp",
    "WeiXena",
    "WesleyYard",
    "WillZen",
    "XavierAnne",
    "YannickBoot",
    "YusufCliff",
    "ZacharyDust",
    "ZaneEpps",
    "AndreAqua",
    "BoBowes",
    "CarlCove",
    "DonDrake",
    "EricEddy",
    "FrankFink",
    "GregGlade",
    "HenryHawk",
    "IgorIdle",
    "JayJasper",
    "KenKnox",
    "LeoLance",
    "MiloMills",
    "NashNile",
    "OttoOwl",
    "PetePlume",
    "QuinnQuill",
)


@dataclass(frozen=True, slots=True)
class AlphabetSortExample:
    """One multi-turn AlphabetSort episode."""

    initial_prompt: str
    follow_ups: list[str]
    ground_truths: list[list[str]]
    turn_names: list[list[str]]
    sort_by_first: bool

    @property
    def num_turns(self) -> int:
        return len(self.turn_names)


def build_example(
    *,
    seed: int,
    step: int,
    group_idx: int,
    min_turns: int,
    max_turns: int,
    min_names_per_turn: int,
    max_names_per_turn: int,
) -> AlphabetSortExample:
    """Build one deterministic episode from local names."""
    rng = random.Random(f"{seed}:{step}:{group_idx}")
    num_turns = rng.randint(min_turns, max_turns)
    names_per_turn = [
        rng.randint(min_names_per_turn, max_names_per_turn) for _ in range(num_turns)
    ]
    names_needed = sum(names_per_turn)
    selected_names = rng.sample(list(NAMES), k=names_needed)
    sort_by_first = rng.choice([True, False])

    turn_names: list[list[str]] = []
    offset = 0
    for count in names_per_turn:
        turn_names.append(selected_names[offset : offset + count])
        offset += count

    ground_truths: list[list[str]] = []
    cumulative: list[str] = []
    for turn_idx, names in enumerate(turn_names):
        cumulative.extend(names)
        sorted_names = sorted(
            cumulative,
            key=_extract_first_name if sort_by_first else _extract_last_name,
        )
        if turn_idx == 0:
            ground_truths.append(sorted_names)
        else:
            ground_truths.append(
                [
                    f"{name} // new name!" if name in names else name
                    for name in sorted_names
                ]
            )

    sort_key = "FIRST" if sort_by_first else "LAST"
    shuffled_first = list(turn_names[0])
    rng.shuffle(shuffled_first)
    first_names = ", ".join(shuffled_first)
    initial_prompt = f"""Sort these names in alphabetical order by {sort_key} name: {first_names}

Use exactly this format:
<alphabetical_sorted>
Name1
Name2
</alphabetical_sorted>"""

    follow_ups = []
    for turn_idx, names in enumerate(turn_names[1:], start=1):
        shuffled = list(names)
        rng.shuffle(shuffled)
        name_list = ", ".join(shuffled)
        new_name_instruction = (
            "These are in addition to the prior list. Mark any NEW names "
            "(that weren't in the prior list) with `// new name!` at the end."
        )
        if turn_idx == 1:
            follow_ups.append(
                f"""New names to add to the prior list: {name_list}

Sort the COMPLETE cumulative list alphabetically by {sort_key} name.

{new_name_instruction}

Use exactly this format:
<combined_alphabetical_sorted>
Name1
Name2 // new name!
</combined_alphabetical_sorted>"""
            )
        else:
            follow_ups.append(
                f"""New names to add to the prior list: {name_list}

Sort the COMPLETE cumulative list alphabetically by {sort_key} name.

{new_name_instruction} Follow the same format as before."""
            )

    return AlphabetSortExample(
        initial_prompt=initial_prompt,
        follow_ups=follow_ups,
        ground_truths=ground_truths,
        turn_names=turn_names,
        sort_by_first=sort_by_first,
    )


def _extract_first_name(name: str) -> str:
    for idx, char in enumerate(name[1:], start=1):
        if char.isupper():
            return name[:idx].lower()
    return name.lower()


def _extract_last_name(name: str) -> str:
    for idx, char in enumerate(name[1:], start=1):
        if char.isupper():
            return name[idx:].lower()
    return ""
