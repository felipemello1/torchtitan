# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opening lines from the Lichess chess-openings dataset (CC0,
https://github.com/lichess-org/chess-openings): for each ECO code, one random line of 8-16 plies; then
200 of those codes sampled at random (seed 0), A00 (irregular first moves) left out."""

# (ECO code and name, moves in SAN)
OPENINGS: tuple[tuple[str, str], ...] = (
    (
        "A02 Bird Opening: From's Gambit, Lipke Variation",
        "f4 e5 fxe5 d6 exd6 Bxd6 Nf3 Nh6 d4",
    ),
    ("A03 Bird Opening: Thomas Gambit", "f4 d5 b3 Nf6 Bb2 d4 Nf3 c5 e3"),
    ("A07 King's Indian Attack: Keres Variation", "Nf3 d5 g3 c6 Bg2 Bg4 O-O Nd7"),
    ("A08 Zukertort Opening: Reversed Grünfeld", "Nf3 d5 g3 c5 Bg2 Nc6 d4 Nf6"),
    (
        "A10 English Opening: King's English Variation, Botvinnik System, Prickly Pawn Pass System",
        "c4 g6 Nc3 Bg7 g3 Nf6 Bg2 O-O e4 d6 Nge2 e5 O-O c6 d3 a6",
    ),
    ("A13 English Opening: Neo-Catalan Declined", "c4 e6 Nf3 d5 g3 Nf6 Bg2 Be7"),
    (
        "A16 English Opening: Anglo-Indian Defense, Anglo-Grünfeld Variation",
        "c4 Nf6 Nc3 d5 cxd5 Nxd5 Nf3 g6",
    ),
    (
        "A19 English Opening: Anglo-Indian Defense, Flohr-Mikenas-Carls Variation, Nei Gambit",
        "c4 e6 Nc3 Nf6 e4 c5 e5 Ng8",
    ),
    (
        "A21 English Opening: King's English Variation, Troger Defense",
        "c4 e5 Nc3 Nc6 g3 d6 Bg2 Be6",
    ),
    (
        "A22 English Opening: King's English Variation, Bellon Gambit",
        "c4 e5 Nc3 Nf6 Nf3 e4 Ng5 b5",
    ),
    (
        "A23 English Opening: King's English Variation, Two Knights Variation, Keres Variation",
        "c4 e5 Nc3 Nf6 g3 Bc5 Bg2 c6",
    ),
    (
        "A25 English Opening: Closed, Taimanov Variation",
        "c4 e5 Nc3 Nc6 g3 g6 Bg2 Bg7 e3 d6 Nge2 Nh6",
    ),
    (
        "A26 English Opening: King's English Variation, Closed System, Full Symmetry",
        "c4 e5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6",
    ),
    (
        "A29 English Opening: King's English Variation, Four Knights Variation, Fianchetto Line, with .. d6, Be7",
        "c4 e5 Nc3 Nf6 g3 Nc6 Bg2 d6 d3 Be7 Nf3 O-O",
    ),
    (
        "A31 English Opening: Symmetrical Variation, Anti-Benoni Variation, Kasparov-Vaganian Gambit",
        "d4 Nf6 c4 c5 Nf3 cxd4 Nxd4 e5 Nb5 d5",
    ),
    (
        "A35 English Opening: Symmetrical Variation, Four Knights Variation, Keres-Parma System",
        "c4 c5 Nf3 Nf6 Nc3 Nc6 g3 e6",
    ),
    (
        "A37 English Opening: Symmetrical Variation, Three Knights, Fianchetto Variation",
        "c4 c5 Nf3 Nc6 g3 g6 Bg2 Bg7 O-O d6 Nc3",
    ),
    (
        "A38 English Opening: Symmetrical Variation, Duchamp Variation",
        "c4 Nf6 Nf3 g6 g3 Bg7 Bg2 O-O O-O c5 Nc3 Nc6 d3",
    ),
    (
        "A39 English Opening: Symmetrical Variation, Mecking Variation",
        "c4 Nf6 Nf3 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 O-O O-O d4",
    ),
    (
        "A40 Benoni Defense: Franco-Sicilian Hybrid",
        "d4 e6 c4 c5 d5 exd5 cxd5 d6 Nc3 g6 e4 Bg7 Nf3 Ne7",
    ),
    (
        "A45 Trompowsky Attack: Raptor Variation, Hergert Gambit",
        "d4 Nf6 Bg5 Ne4 h4 Nxg5 hxg5 e5",
    ),
    ("A47 Pseudo Queen's Indian Defense", "d4 Nf6 Nf3 b6 e3 Bb7 Bd3 e6 Nbd2 c5 b3 Be7"),
    (
        "A48 Queen's Pawn Game: Barry Attack, Grünfeld Variation",
        "d4 Nf6 Nf3 g6 Nc3 d5 Bf4 Bg7 e3 O-O Be2",
    ),
    (
        "A52 Indian Defense: Budapest Gambit Accepted, Main Line, Alekhine Variation, Tartakower Defense",
        "d4 Nf6 c4 e5 dxe5 Ng4 e4 d6",
    ),
    (
        "A54 Old Indian Defense: Duz-Khotimirsky Variation",
        "d4 Nf6 c4 d6 Nc3 e5 e3 Nbd7 Bd3",
    ),
    ("A56 Benoni Defense: King's Indian System", "d4 Nf6 c4 c5 d5 e5 Nc3 d6 e4 g6"),
    (
        "A59 Benko Gambit Accepted: Yugoslav",
        "d4 Nf6 c4 c5 d5 b5 cxb5 a6 bxa6 Bxa6 Nc3 d6 e4",
    ),
    (
        "A60 Benoni Defense: Modern Variation, Snake Variation",
        "d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 Bd6",
    ),
    ("A65 Benoni Defense: King's Pawn Line", "d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6 e4"),
    (
        "A66 Benoni Defense: Pawn Storm Variation",
        "d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6 e4 g6 f4",
    ),
    (
        "A67 Benoni Defense: Taimanov Variation",
        "d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6 e4 g6 f4 Bg7 Bb5+",
    ),
    (
        "A68 Benoni Defense: Four Pawns Attack",
        "d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6 e4 g6 f4 Bg7 Nf3 O-O",
    ),
    (
        "A72 Benoni Defense: Classical Variation",
        "d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6 e4 g6 Nf3 Bg7 Be2 O-O",
    ),
    (
        "A83 Dutch Defense: Staunton Gambit, Chigorin Variation",
        "d4 f5 e4 fxe4 Nc3 Nf6 Bg5 c6",
    ),
    ("A84 Dutch Defense: Krause Variation", "d4 f5 c4 Nf6 Nc3 d6 Nf3 Nc6"),
    (
        "A88 Dutch Defense: Leningrad Variation, Warsaw Variation",
        "d4 f5 g3 Nf6 Bg2 g6 Nf3 Bg7 O-O O-O c4 d6 Nc3 c6",
    ),
    (
        "A89 Dutch Defense: Leningrad Variation, Matulovic Variation",
        "d4 f5 g3 Nf6 Bg2 g6 Nf3 Bg7 O-O O-O c4 d6 Nc3 Nc6",
    ),
    (
        "A91 Dutch Defense: Classical Variation, Blackburne Attack",
        "d4 f5 c4 Nf6 g3 e6 Bg2 Be7 Nh3",
    ),
    ("A92 Dutch Defense: Classical Variation", "d4 f5 c4 Nf6 g3 e6 Bg2 Be7 Nf3 O-O"),
    (
        "A93 Dutch Defense: Stonewall Variation, Botvinnik Variation",
        "d4 f5 c4 Nf6 g3 e6 Bg2 Be7 Nf3 O-O O-O d5 b3",
    ),
    (
        "A98 Dutch Defense: Classical Variation, Ilyin-Zhenevsky Variation, Alatortsev-Lisitsyn Line",
        "d4 f5 c4 Nf6 g3 e6 Bg2 Be7 Nf3 O-O O-O d6 Nc3 Qe8 Qc2",
    ),
    (
        "B00 Nimzowitsch Defense: El Columpio Defense, Exchange Variation",
        "e4 Nc6 Nf3 Nf6 e5 Ng4 d4 d6 h3 Nh6 exd6",
    ),
    (
        "B01 Scandinavian Defense: Schiller-Pytel Variation, Modern Variation",
        "e4 d5 exd5 Qxd5 Nc3 Qd6 d4 Nf6 Bc4 c6 Nge2 Bf5 Bf4 Qb4",
    ),
    (
        "B02 Alekhine Defense: Hunt Variation, Matsukevich Gambit",
        "e4 Nf6 e5 Nd5 c4 Nb6 c5 Nd5 Nc3 Nxc3 dxc3 d6 Bg5",
    ),
    (
        "B03 Alekhine Defense: Four Pawns Attack, Cambridge Gambit",
        "e4 Nf6 e5 Nd5 d4 d6 c4 Nb6 f4 g5",
    ),
    (
        "B05 Alekhine Defense: Modern Variation, Alekhine Variation",
        "e4 Nf6 e5 Nd5 d4 d6 Nf3 Bg4 c4",
    ),
    (
        "B08 Pirc Defense: Classical Variation, Quiet System, Parma Defense",  # codespell:ignore
        "e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7 Be2 O-O O-O Bg4",
    ),
    (
        "B09 Pirc Defense: Austrian Attack, Unzicker Attack, Bronstein Variation",
        "e4 d6 d4 Nf6 Nc3 g6 f4 Bg7 Nf3 O-O e5 Nfd7 h4",
    ),
    (
        "B17 Caro-Kann Defense: Karpov Variation, Smyslov Variation",
        "e4 c6 d4 d5 Nd2 dxe4 Nxe4 Nd7 Bc4 Ngf6 Ng5 e6 Qe2 Nb6",
    ),
    (
        "B19 Caro-Kann Defense: Classical Variation, Spassky Variation",
        "e4 c6 d4 d5 Nd2 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5",
    ),
    (
        "B20 Sicilian Defense: Wing Gambit, Romanian Defense",
        "e4 c5 b4 cxb4 a3 d5 exd5 Qxd5 Nf3 e5 Bb2 Nc6 c4 Qe6",
    ),
    ("B22 Sicilian Defense: Heidenfeld Variation", "e4 c5 c3 Nf6 e5 Nd5 Nf3 Nc6 Na3"),
    (
        "B23 Sicilian Defense: Closed, Carlsen Variation",
        "e4 c5 Nc3 d6 d4 cxd4 Qxd4 Nc6 Qd2",
    ),
    (
        "B25 Sicilian Defense: Closed, Botvinnik Defense, Edge Variation",
        "e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6 f4 e5 Nh3 Nge7",
    ),
    (
        "B31 Sicilian Defense: Nyezhmetdinov-Rossolimo Attack, Fianchetto Variation, Gufeld Gambit",
        "e4 c5 Nf3 Nc6 Bb5 g6 O-O Bg7 c3 e5 d4",
    ),
    (
        "B36 Sicilian Defense: Accelerated Dragon, Maróczy Bind, Gurgenidze Variation",
        "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6 c4 Nf6 Nc3 Nxd4 Qxd4 d6",
    ),
    ("B41 Sicilian Defense: Kan Variation", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6"),
    (
        "B42 Sicilian Defense: Kan Variation, Modern Variation",
        "e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Bd3",
    ),
    (
        "B43 Sicilian Defense: Kan Variation, Knight Variation",
        "e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Nc3 b5 g3 Bb7 Bg2",
    ),
    ("B44 Sicilian Defense: Taimanov Variation", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6"),
    (
        "B45 Sicilian Defense: Four Knights Variation, Sveshnikov Transfer",
        "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Nf6 Ndb5 d6",
    ),
    (
        "B48 Sicilian Defense: Taimanov Variation, Bastrikov Variation, English Attack",
        "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7 Be3",
    ),
    (
        "B55 Sicilian Defense: Prins Variation, Venice Attack",
        "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 f3 e5 Bb5+",
    ),
    (
        "B56 Sicilian Defense: Venice Attack",
        "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Bb5+",
    ),
    (
        "B60 Sicilian Defense: Richter-Rauzer Variation, Modern Variation",
        "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bg5 Bd7",
    ),
    (
        "B80 Sicilian Defense: Scheveningen Variation, English Attack, with Qd2",
        "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e6 Qd2",
    ),
    (
        "B82 Sicilian Defense: Scheveningen Variation, Tal Variation",
        "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6 f4 Nc6 Be3 Be7 Qf3",
    ),
    (
        "B84 Sicilian Defense: Scheveningen Variation, Classical Variation",
        "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e6",
    ),
    (
        "B85 Sicilian Defense: Scheveningen Variation, Classical Variation, Paulsen Variation",
        "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 f4 e6 Be2 Qc7 O-O Nc6",
    ),
    ("B86 Sicilian Defense: Sozin Attack", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 d6 Bc4"),
    (
        "B87 Sicilian Defense: Sozin Attack, Flank Variation",
        "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bc4 e6 Bb3 b5",
    ),
    (
        "B88 Sicilian Defense: Sozin Attack, Leonhardt Variation",
        "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 d6 Bc4 e6",
    ),
    (
        "B89 Sicilian Defense: Sozin Attack, Main Line",
        "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 d6 Bc4 e6 Be3",
    ),
    (
        "B90 Sicilian Defense: Najdorf Variation, Yates Variation",
        "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bd3",
    ),
    ("C00 French Defense: Hoffmann Gambit", "e4 e6 d4 d5 Qe2 e5 f4 exf4"),
    (
        "C01 French Defense: Exchange Variation, Bogoljubow Variation",
        "e4 e6 d4 d5 exd5 exd5 Nc3 Nf6 Bg5 Nc6",
    ),
    (
        "C03 French Defense: Tarrasch Variation, Guimard Defense, Thunderbunny Variation",
        "e4 e6 d4 d5 Nd2 Nc6 c3 dxe4 Nxe4 e5",
    ),
    (
        "C04 French Defense: Tarrasch Variation, Guimard Defense, Main Line",
        "e4 e6 d4 d5 Nd2 Nc6 Ngf3 Nf6",
    ),
    (
        "C05 French Defense: Tarrasch Variation, Botvinnik Variation",
        "e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7 Bd3 c5 c3 b6",
    ),
    (
        "C06 French Defense: Tarrasch Variation, Leningrad Variation",
        "e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7 Bd3 c5 c3 Nc6 Ne2 cxd4 cxd4 Nb6",
    ),
    (
        "C07 French Defense: Tarrasch Variation, Open System, Shaposhnikov Gambit",
        "e4 e6 d4 d5 Nd2 c5 exd5 Nf6",
    ),
    (
        "C11 French Defense: Steinitz Variation, Boleslavsky Variation",
        "e4 e6 d4 d5 Nc3 Nf6 e5 Nfd7 f4 c5 Nf3 Nc6 Be3",
    ),
    (
        "C12 French Defense: McCutcheon Variation, Lasker Variation",
        "e4 e6 d4 d5 Nc3 Nf6 Bg5 Bb4 e5 h6 Bd2 Bxc3 bxc3 Ne4 Qg4 g6",
    ),
    (
        "C13 French Defense: Classical Variation, Burn Variation, Morozevich Line",
        "e4 e6 d4 d5 Nc3 Nf6 Bg5 dxe4 Nxe4 Be7 Bxf6 gxf6",
    ),
    (
        "C15 French Defense: Winawer Variation, Fingerslip Variation, Kunin Double Gambit",
        "e4 e6 d4 d5 Nc3 Bb4 Bd2 dxe4 Qg4 Qxd4",
    ),
    (
        "C17 French Defense: Winawer Variation, Bogoljubow Variation",
        "e4 e6 d4 d5 Nc3 Bb4 e5 c5 Bd2",
    ),
    (
        "C18 French Defense: Winawer Variation, Advance Variation, with Bd3",
        "e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+ bxc3 Ne7 Bd3",
    ),
    ("C22 Center Game: Charousek Variation", "e4 e5 d4 exd4 Qxd4 Nc6 Qe3 Bb4+ c3 Be7"),
    (
        "C25 Vienna Gambit, with Max Lange Defense: Hamppe-Allgaier Gambit",
        "e4 e5 Nc3 Nc6 f4 exf4 Nf3 g5 h4 g4 Ng5",
    ),
    (
        "C26 Vienna Game: Stanley Variation, Eifel Gambit",
        "e4 e5 Nc3 Nf6 Bc4 Bc5 Nge2 b5",
    ),
    (
        "C30 King's Gambit Declined: Panteldakis Countergambit, Pawn Sacrifice Line",
        "e4 e5 f4 f5 exf5 exf4 Qh5+ g6 fxg6 Qe7+ Kd1",
    ),
    (
        "C31 King's Gambit Declined: Falkbeer Countergambit, Pickler Gambit",
        "e4 e5 f4 d5 exd5 c6 dxc6 Bc5",
    ),
    (
        "C32 King's Gambit Declined: Falkbeer Countergambit, Charousek Gambit, Old Line",
        "e4 e5 f4 d5 exd5 e4 d3 Nf6 Qe2",
    ),
    (
        "C33 King's Gambit Accepted: Bishop's Gambit, Cozio Variation",
        "e4 e5 f4 exf4 Bc4 Qh4+ Kf1 d6",
    ),
    (
        "C35 King's Gambit Accepted: Cunningham Defense, Bertin Gambit",
        "e4 e5 f4 exf4 Nf3 Be7 Bc4 Bh4+ g3 fxg3 O-O gxh2+ Kh1",
    ),
    (
        "C43 Bishop's Opening: Urusov Gambit, Keidansky Gambit",
        "e4 e5 Bc4 Nf6 d4 exd4 Nf3 Nxe4 Qxd4",
    ),
    (
        "C45 Scotch Game: Romanishin Variation",
        "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Nb3 Bb4+",
    ),
    (
        "C47 Four Knights Game: Scotch Variation, Belgrade Gambit",
        "e4 e5 Nf3 Nc6 Nc3 Nf6 d4 exd4 Nd5",
    ),
    (
        "C48 Four Knights Game: Spanish Variation, Rubinstein Variation",
        "e4 e5 Nf3 Nc6 Nc3 Nf6 Bb5 Nd4",
    ),
    ("C50 Italian Game: Giuoco Pianissimo", "e4 e5 Nf3 Nc6 Bc4 Nf6 O-O Bc5 Nc3 d6 d3"),
    (
        "C51 Italian Game: Evans Gambit, McDonnell Defense",
        "e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Bc5",
    ),
    (
        "C53 Italian Game: Classical Variation, Mestel Variation",
        "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Qe7 d4 Bb6 Bg5",
    ),
    (
        "C54 Italian Game: Classical Variation, Greco Gambit, Dubov Italian",
        "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d4 exd4 b4",
    ),
    (
        "C55 Italian Game: Two Knights Defense",
        "e4 e5 Nf3 Nc6 Bc4 Nf6 O-O Bc5 d4 Bxd4 Nxd4 Nxd4 Bg5 d6",
    ),
    (
        "C62 Ruy Lopez: Steinitz Defense, Center Gambit",
        "e4 e5 Nf3 Nc6 Bb5 d6 d4 exd4 O-O",
    ),
    (
        "C63 Ruy Lopez: Schliemann Defense, Tartakower Variation",
        "e4 e5 Nf3 Nc6 Bb5 f5 Nc3 fxe4 Nxe4 Nf6",
    ),
    (
        "C64 Ruy Lopez: Classical Defense, Benelux Variation",
        "e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Bc5 c3 O-O d4 Bb6",
    ),
    (
        "C65 Ruy Lopez: Berlin Defense, Anti-Berlin Variation, Mortimer Variation",
        "e4 e5 Nf3 Nc6 Bb5 Nf6 d3 Ne7",
    ),
    (
        "C68 Ruy Lopez: Exchange Variation, Romanovsky Variation",
        "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 Nc3 f6 d3",
    ),
    (
        "C69 Ruy Lopez: Exchange Variation, Normal Variation",
        "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 O-O",
    ),
    (
        "C70 Ruy Lopez: Morphy Defense, Norwegian Variation",
        "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 b5 Bb3 Na5",
    ),
    (
        "C73 Ruy Lopez: Morphy Defense, Modern Steinitz Defense",
        "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 d6 Bxc6+ bxc6 d4 f6",
    ),
    (
        "C74 Ruy Lopez: Morphy Defense, Modern Steinitz Defense",
        "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 d6 c3 f5 exf5 Bxf5 O-O",
    ),
    (
        "C75 Ruy Lopez: Morphy Defense, Modern Steinitz Defense",
        "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 d6 c3 Bd7 d4 Nge7",
    ),
    (
        "C76 Ruy Lopez: Morphy Defense, Modern Steinitz Defense, Fianchetto Variation",
        "e4 e5 Nf3 Nc6 Bb5 g6 c3 a6 Ba4 d6 d4 Bd7",
    ),
    (
        "C84 Ruy Lopez: Closed, Morphy Attack",
        "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Nc3",
    ),
    (
        "C86 Ruy Lopez: Closed, Worrall Attack, Delayed Castling Line",
        "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Qe2 b5 Bb3 d6",
    ),
    (
        "C89 Ruy Lopez: Marshall Attack",
        "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 O-O c3 d5",
    ),
    (
        "C90 Ruy Lopez: Closed",
        "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O",
    ),
    (
        "D01 Richter-Veresov Attack: Malich Gambit",
        "d4 Nf6 Nc3 d5 Bg5 c5 Bxf6 gxf6 e4 dxe4 d5",
    ),
    ("D02 Queen's Pawn Game: London System", "d4 d5 Nf3 Nf6 Bf4 c5 e3 Nc6 Nbd2"),
    ("D03 Queen's Pawn Game: Torre Attack", "d4 d5 Nf3 Nf6 Bg5 e6 e3 c5 Nbd2"),
    (
        "D04 Queen's Pawn Game: Colle System, Grünfeld Formation",
        "d4 d5 Nf3 Nf6 e3 g6 Bd3 Bg7",
    ),
    (
        "D05 Rubinstein Opening: Semi-Slav Defense",
        "d4 d5 Nf3 Nf6 e3 e6 Bd3 Bd6 O-O O-O b3 Nbd7 Bb2 c6",
    ),
    (
        "D06 Queen's Gambit Declined: Baltic Defense, Pseudo-Chigorin",
        "d4 d5 c4 Bf5 Nc3 e6 Nf3 Nc6",
    ),
    (
        "D08 Queen's Gambit Declined: Albin Countergambit, Janowski Variation",
        "d4 d5 c4 e5 dxe5 d4 Nf3 Nc6 Nbd2 f6",
    ),
    ("D11 Slav Defense: Modern Line", "d4 d5 Nf3 Nf6 g3 c6 Bg2 Bg4 O-O Nbd7 c4"),
    (
        "D12 Slav Defense: Quiet Variation, Schallopp Defense",
        "d4 d5 c4 c6 Nf3 Nf6 e3 Bf5",
    ),
    ("D13 Slav Defense: Exchange Variation", "d4 d5 c4 c6 Nf3 Nf6 cxd5 cxd5"),
    ("D15 Slav Defense: Süchting Variation", "d4 d5 c4 c6 Nf3 Nf6 Nc3 Qb6"),
    (
        "D18 Slav Defense: Czech Variation, Classical System",
        "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3",
    ),
    (
        "D22 Queen's Gambit Accepted: Alekhine Defense, Haberditz Variation",
        "d4 d5 c4 dxc4 Nf3 a6 e3 b5",
    ),
    (
        "D23 Queen's Gambit Accepted: Mannheim Variation",
        "d4 d5 c4 dxc4 Nf3 Nf6 Qa4+ c6 Qxc4",
    ),
    (
        "D28 Queen's Gambit Accepted: Classical Defense, Alekhine System",
        "d4 d5 c4 dxc4 Nf3 Nf6 e3 e6 Bxc4 c5 O-O a6 Qe2 b5",
    ),
    (
        "D29 Queen's Gambit Accepted: Classical Defense, Alekhine System, Main Line",
        "d4 d5 c4 dxc4 Nf3 Nf6 e3 e6 Bxc4 c5 O-O a6 Qe2 b5 Bb3 Bb7",
    ),
    (
        "D39 Queen's Gambit Declined: Ragozin Defense, Vienna Variation",
        "d4 Nf6 c4 e6 Nf3 d5 Nc3 Bb4 Bg5 dxc4",
    ),
    (
        "D40 Queen's Gambit Declined: Semi-Tarrasch Defense, with e3",
        "d4 d5 c4 c5 Nf3 Nf6 Nc3 e6 e3 Nc6 Bd3",
    ),
    (
        "D41 Queen's Gambit Declined: Semi-Tarrasch Defense",
        "c4 c5 Nf3 Nf6 Nc3 Nc6 g3 d5 d4 e6",
    ),
    (
        "D43 Semi-Slav Defense: Moscow Variation",
        "d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 Bg5 h6 Bxf6 Qxf6",
    ),
    (
        "D48 Semi-Slav Defense: Meran Variation",
        "d4 d5 c4 c6 Nc3 Nf6 e3 e6 Nf3 Nbd7 Bd3 dxc4 Bxc4 b5 Bd3 a6",
    ),
    (
        "D55 Queen's Gambit Declined: Modern Variation, Normal Line",
        "d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 O-O e3",
    ),
    (
        "D56 Queen's Gambit Declined: Lasker Defense",
        "d4 d5 c4 e6 Nc3 Be7 Nf3 Nf6 Bg5 h6 Bh4 O-O e3 Ne4",
    ),
    (
        "D58 Queen's Gambit Declined: Tartakower Defense, Exchange Variation",
        "d4 d5 c4 e6 Nc3 Be7 Nf3 Nf6 Bg5 h6 Bh4 O-O e3 b6 cxd5 exd5",
    ),
    (
        "D61 Queen's Gambit Declined: Orthodox Defense, Rubinstein Variation",
        "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 Nbd7 Qc2",
    ),
    (
        "D62 Queen's Gambit Declined: Orthodox Defense, Rubinstein Variation, Flohr Line",
        "d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 O-O e3 Nbd7 Qc2 c5 cxd5",
    ),
    (
        "D63 Queen's Gambit Declined: Orthodox Defense, Swiss, Carlsbad Variation",
        "d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 O-O e3 Nbd7 Rc1 a6 cxd5",
    ),
    (
        "D64 Queen's Gambit Declined: Orthodox Defense, Rubinstein Attack",
        "Nf3 d5 d4 Nf6 c4 e6 Nc3 Be7 Bg5 O-O e3 Nbd7 Rc1 c6 Qc2 Ne4",
    ),
    (
        "D71 Neo-Grünfeld Defense: Exchange Variation",
        "d4 Nf6 c4 g6 g3 Bg7 Bg2 d5 cxd5 Nxd5",
    ),
    ("D73 Neo-Grünfeld Defense: with g3", "d4 Nf6 c4 g6 g3 d5 Bg2 Bg7 Nf3"),
    (
        "D75 Neo-Grünfeld Defense: Delayed Exchange Variation",
        "d4 Nf6 Nf3 g6 c4 Bg7 g3 O-O Bg2 d5 cxd5 Nxd5 O-O c5 dxc5",
    ),
    (
        "D77 Neo-Grünfeld Defense: Classical Variation, Modern Defense",
        "d4 Nf6 Nf3 g6 g3 Bg7 Bg2 O-O O-O d5 c4 dxc4",
    ),
    (
        "D78 Neo-Grünfeld Defense: Classical Variation, Original Defense",
        "d4 Nf6 c4 g6 Nf3 Bg7 g3 O-O Bg2 c6 O-O d5",
    ),
    ("D85 Grünfeld Defense: Exchange Variation", "d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5"),
    (
        "D87 Grünfeld Defense: Exchange Variation, Spassky Variation",
        "d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Bc4 c5 Ne2 O-O",
    ),
    ("D90 Grünfeld Defense: Flohr Variation", "d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qa4+"),
    (
        "D94 Grünfeld Defense: Opocensky Variation",
        "d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 e3 O-O Bd2",
    ),
    (
        "D95 Grünfeld Defense: Pachman Variation",
        "d4 Nf6 c4 g6 Nc3 d5 e3 Bg7 Qb3 dxc4 Bxc4 O-O Nf3 Nbd7 Ng5",
    ),
    (
        "E05 Catalan Opening: Open Defense, Classical Line",
        "d4 Nf6 c4 e6 Nf3 d5 g3 Be7 Bg2 O-O O-O dxc4",
    ),
    (
        "E07 Catalan Opening: Closed, Botvinnik Variation",
        "d4 Nf6 c4 e6 g3 d5 Bg2 Be7 Nf3 O-O O-O Nbd7 Nc3 c6 Qd3",
    ),
    (
        "E08 Catalan Opening: Closed",
        "d4 Nf6 c4 e6 g3 d5 Bg2 Be7 Nf3 O-O O-O Nbd7 Qc2 c6 b3",
    ),
    (
        "E10 Blumenfeld Countergambit: Duz-Khotimirsky Variation",
        "d4 Nf6 c4 e6 Nf3 c5 d5 b5 Bg5",
    ),
    (
        "E11 Bogo-Indian Defense: Retreat Variation",
        "d4 Nf6 c4 e6 g3 Bb4+ Bd2 Be7 Bg2 d5 Nf3 O-O",
    ),
    (
        "E15 Queen's Indian Defense: Fianchetto Variation, Traditional Line",
        "d4 Nf6 c4 e6 Nf3 b6 g3 Bb7",
    ),
    (
        "E17 Queen's Indian Defense: Classical Variation",
        "d4 Nf6 c4 e6 Nf3 b6 g3 Bb7 Bg2 Be7 O-O",
    ),
    (
        "E20 Nimzo-Indian Defense: Romanishin Variation",
        "d4 Nf6 c4 e6 Nc3 Bb4 Nf3 c5 g3 O-O Bg2",
    ),
    (
        "E21 Nimzo-Indian Defense: Three Knights Variation, Euwe Variation",
        "d4 Nf6 c4 e6 Nc3 Bb4 Nf3 c5 d5 Ne4",
    ),
    (
        "E25 Nimzo-Indian Defense: Sämisch Variation",
        "d4 Nf6 c4 e6 Nc3 Bb4 f3 d5 a3 Bxc3+ bxc3 c5 cxd5",
    ),
    (
        "E26 Nimzo-Indian Defense: Sämisch Variation, O'Kelly Variation",
        "d4 Nf6 c4 e6 Nc3 Bb4 a3 Bxc3+ bxc3 c5 e3 b6",
    ),
    (
        "E29 Nimzo-Indian Defense: Sämisch Variation",
        "d4 Nf6 c4 e6 Nc3 Bb4 e3 c5 Bd3 Nc6 a3 Bxc3+ bxc3 O-O",
    ),
    (
        "E30 Nimzo-Indian Defense: Leningrad Variation, Averbakh Gambit",
        "d4 Nf6 c4 e6 Nc3 Bb4 Bg5 h6 Bh4 c5 d5 b5",
    ),
    (
        "E31 Nimzo-Indian Defense: Leningrad Variation, Benoni Defense",
        "d4 Nf6 c4 e6 Nc3 Bb4 Bg5 h6 Bh4 c5 d5 d6",
    ),
    (
        "E35 Nimzo-Indian Defense: Classical Variation, Noa Variation",
        "d4 Nf6 c4 e6 Nc3 Bb4 Qc2 d5 cxd5 exd5",
    ),
    (
        "E38 Nimzo-Indian Defense: Classical Variation, Berlin Variation",
        "d4 Nf6 c4 e6 Nc3 Bb4 Qc2 c5",
    ),
    (
        "E39 Nimzo-Indian Defense: Classical Variation, Berlin Variation, Pirc Variation",
        "d4 Nf6 c4 e6 Nc3 Bb4 Qc2 c5 dxc5 O-O",
    ),
    (
        "E40 Nimzo-Indian Defense: Rubinstein System, Taimanov Variation",
        "d4 Nf6 c4 e6 Nc3 Bb4 e3 Nc6",
    ),
    (
        "E41 Nimzo-Indian Defense: Rubinstein System, Hübner Variation",
        "d4 Nf6 c4 e6 Nc3 Bb4 e3 c5 Bd3 Nc6 Nf3 Bxc3+ bxc3 d6",
    ),
    (
        "E42 Nimzo-Indian Defense: Rubinstein System, Rubinstein Variation",
        "d4 Nf6 c4 e6 Nc3 Bb4 e3 c5 Ne2",
    ),
    (
        "E44 Nimzo-Indian Defense: St. Petersburg Variation",
        "d4 Nf6 c4 e6 Nc3 Bb4 e3 b6 Ne2",
    ),
    (
        "E45 Nimzo-Indian Defense: St. Petersburg Variation, Fischer Variation",
        "d4 Nf6 c4 e6 Nc3 Bb4 e3 b6 Ne2 Ba6",
    ),
    (
        "E46 Nimzo-Indian Defense: Simagin Variation",
        "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Ne2 d5 a3 Bd6",
    ),
    (
        "E49 Nimzo-Indian Defense: Normal Variation, Botvinnik System",
        "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 a3 Bxc3+ bxc3",
    ),
    ("E50 Nimzo-Indian Defense", "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Nf3"),
    (
        "E51 Nimzo-Indian Defense: Normal Variation, Sämisch Deferred",
        "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Nf3 d5 a3",
    ),
    (
        "E52 Nimzo-Indian Defense: Normal Variation, Schlechter Defense",
        "d4 Nf6 c4 e6 Nc3 Bb4 e3 b6 Bd3 Bb7 Nf3 O-O O-O d5",
    ),
    (
        "E53 Nimzo-Indian Defense: Normal Variation, Gligoric System",
        "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3 c5 O-O Nbd7",
    ),
    (
        "E54 Nimzo-Indian Defense: Normal Variation, Gligoric System, Exchange Variation",
        "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3 c5 O-O dxc4 Bxc4",
    ),
    (
        "E61 King's Indian Defense: Semi-Classical Variation",
        "d4 Nf6 c4 g6 Nc3 Bg7 Nf3 O-O e3 d6 Be2",
    ),
    (
        "E65 King's Indian Defense: Fianchetto Variation, Yugoslav Variation, Exchange Line",
        "d4 Nf6 c4 g6 Nf3 Bg7 g3 O-O Bg2 d6 O-O c5 Nc3 Nc6 dxc5 dxc5",
    ),
    (
        "E66 King's Indian Defense: Fianchetto Variation, Yugoslav Variation, Advance Line",
        "d4 Nf6 c4 g6 Nf3 Bg7 g3 O-O Bg2 d6 O-O c5 Nc3 Nc6 d5",
    ),
    (
        "E72 King's Indian Defense: Normal Variation, Deferred Fianchetto",
        "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 g3",
    ),
    (
        "E73 King's Indian Defense: Averbakh Variation, Spanish Defense",
        "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Bg5 a6",
    ),
    (
        "E76 King's Indian Defense: Four Pawns Attack, Modern Defense",
        "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f4 Na6",
    ),
    (
        "E81 King's Indian Defense: Sämisch Variation, Byrne Defense",
        "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 c6 Bd3 a6",
    ),
    (
        "E85 King's Indian Defense: Sämisch Variation, Orthodox Variation",
        "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5",
    ),
    (
        "E88 King's Indian Defense: Sämisch Variation, Closed Variation",
        "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 d5 c6",
    ),
    (
        "E91 King's Indian Defense: Kazakh Variation",
        "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 Na6",
    ),
    (
        "E92 King's Indian Defense: Orthodox Variation, Gligoric-Taimanov System",
        "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 Be3",
    ),
    (
        "E94 King's Indian Defense: Orthodox Variation, Positional Defense",
        "d4 Nf6 c4 d6 Nc3 Nbd7 e4 e5 Nf3 g6 Be2 Bg7 O-O O-O",
    ),
    (
        "E95 King's Indian Defense: Orthodox Variation",
        "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nbd7 Re1",
    ),
    (
        "E97 King's Indian Defense: Orthodox Variation, Aronin-Taimanov Defense",
        "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6",
    ),
)
