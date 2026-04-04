'''
Ez a program egy Alice-Bob-Eve típusú neurális kommunikációs modellt valósít meg.

Alice egy bináris üzenet, egy kulcs és egy publikus nonce alapján titkosított reprezentációt állít elő.
Bob a titkosított reprezentáció, a helyes kulcs és a nonce segítségével próbálja rekonstruálni az üzenetet,
míg Eve ugyanezt a feladatot végzi kulcs nélkül, de a nonce ismeretében.

A modell elsődleges célja annak vizsgálata, hogy a neurális hálózatok mennyire képesek
alkalmazkodni ismeretlen és változó jelsorozatokhoz, illetve hogy a dekódoló hálózat
mennyire marad működőképes a jelstruktúra változásai mellett.

A nonce bevezetésével a rendszer nem determinisztikusan viselkedik:
ugyanaz az üzenet és kulcs különböző nonce-ok mellett eltérő cipher reprezentációkhoz vezethet.

A tréning három elkülönített részből áll:
1. Eve tanítása a cipherből történő kulcs nélküli visszafejtésre,
2. Bob tanítása a cipher, a helyes kulcs és a nonce alapján történő dekódolásra,
3. Alice tanítása olyan reprezentáció előállítására, amely Bob számára jól dekódolható,
   Eve számára viszont nehezebben értelmezhető.

A Bob tréningje opcionálisan tartalmazhat egy wrong-nonce kontrollt is

A program elsősorban több seed melletti futtatást támogat, és a végén statisztikai összesítést,
eloszlásvizsgálatot, valamint grafikus megjelenítést készít az eredményekről.
'''

# ============================================================
# IMPORTOK
# ============================================================

import random
import csv
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

# ============================================================
# KONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MSG_BITS = 32       # az üzenet hosszúsága
KEY_BITS = 32       # a kulcs hosszúsága
NONCE_BITS = 32     # a publikus nonce hossza

LAMBDA_EVE = 1.0    # Alice törekvése Eve randomizálására

# Bob wrong-nonce loss súlya
LAMBDA_NONCE = 0.5
LAMBDA_NONCE_ALICE = 0.1
USE_NONCE_LOSS_ALICE = True

# attribútumok különböző tesztekhez
USE_EVE_LOSS = True
USE_RAMP = True
USE_NONCE_LOSS = True

# globális iteráción belüli frissülések számának meghatározása
ALICE_STEPS_PER_ITER = 1
BOB_STEPS_PER_ITER = 2

# ============================================================
# ADATGENERÁLÁS
# ============================================================

def generate_batch(batch_size=128, device=DEVICE):
    msg = torch.randint(0, 2, (batch_size, MSG_BITS), device=device).float()
    key = torch.randint(0, 2, (batch_size, KEY_BITS), device=device).float()
    nonce = torch.randint(0, 2, (batch_size, NONCE_BITS), device=device).float()
    return msg, key, nonce

# ============================================================
# MODELLEK (ALICE, BOB, EVE)
# ============================================================

class Alice(nn.Module):
    """
    Alice
    Bemenet: random generált üzenet, kulcs és nonce
    Kimenet: egy kódolt reprezentáció logitjai
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MSG_BITS + KEY_BITS + NONCE_BITS, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, MSG_BITS)
        )

    def forward(self, msg, key, nonce):
        x = torch.cat([msg, key, nonce], dim=1)
        return self.net(x)


class Bob(nn.Module):
    """
    Bob
    Bemenet: titkosított üzenet, kulcs és nonce
    Kimenet: az eredeti üzenet becsült bitje (csak logitok)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MSG_BITS + KEY_BITS + NONCE_BITS, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, MSG_BITS)
        )

    def forward(self, cipher, key, nonce):
        x = torch.cat([cipher, key, nonce], dim=1)
        return self.net(x)


class Eve(nn.Module):
    """
    Eve
    Bemenet: titkosított üzenet és publikus nonce
    Kimenet: az eredeti üzenet becsült bitje (csak logitok)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MSG_BITS + NONCE_BITS, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, MSG_BITS)
        )

    def forward(self, cipher, nonce):
        x = torch.cat([cipher, nonce], dim=1)
        return self.net(x)

# ============================================================
# SEGÉDLETEK
# ============================================================

'''
Bináris keresztentrópia veszteségfüggvény, ami bináris célértékek esetén méri a modell hibáját
Logitokat használ valós bitek mellett, a sigmoidot automatikusan tartalmazza
'''
bce = nn.BCEWithLogitsLoss()

'''
A több seed-en futtatott kísérletek eredményeiből számol leíró statisztikákat (átlag, szórás, kvartilis, szélsőérték)
'''
def descriptive_stats(values):
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": arr.mean(),
        "std": arr.std(),
        "min": arr.min(),
        "q25": np.percentile(arr, 25),
        "median": np.median(arr),
        "q75": np.percentile(arr, 75),
        "max": arr.max(),
    }

'''
A multi-seed futtatások eredményeit egy csv kiterjesztésű fájlba menti
'''
def save_multi_seed_results(results, filename="multi_seed_results.csv"):
    if not results:
        return

    keys = results[0].keys()

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[INFO] Results saved to: {filename}")

'''
Beállítja az összes random generátor seed-jét (Python, NumPy, PyTorch), hogy a futtatások reprodukálhatóak legyenek
'''
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

'''
Bitenkénti pontosság kiszámítása a modell kimenete és a valódi célérték között
'''
@torch.no_grad()
def bitwise_accuracy_from_logits(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float()
    return (pred == target).float().mean().item()

'''
Mintánkénti átlagos hibás bitek számának kiszámítása
'''
@torch.no_grad()
def hard_bit_error_from_logits(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float()
    return (pred != target).float().sum(dim=1).mean().item()

'''
Lehetővé teszi a modell paramétereinek befagyasztását vagy engedélyezését a tanítás során
'''
def set_requires_grad(model, flag: bool):
    for p in model.parameters():
        p.requires_grad = flag

'''
Alice kimenetéből előállítja a tényleges titkosított reprezentációt a megadott üzenet,
kulcs és publikus nonce alapján.
A neurális hálózat nyers kimenetét hiperbolikus tangens aktivációval [-1, 1] tartományba
korlátozza, így egy folytonos értékű cipher vektort hoz létre.
A függvény ebben a változatban nem használ külön maszkot:
a végső cipher közvetlenül az Alice által előállított kódolt reprezentáció tanh-val
transzformált változata.
'''
def alice_build_cipher(alice, msg, key, nonce):
    enc_logits = alice(msg, key, nonce)
    cipher = torch.tanh(enc_logits)
    return cipher

'''
Egy adott metrika átlagát és szórását számolja ki több seed alapján
'''
def mean_std(values):
    arr = np.array(values, dtype=np.float64)
    return arr.mean(), arr.std()

'''
Több seed-en végrehajtott kísérletek eredményeit összesíti és jeleníti meg
Egyes metrikákra kiszámítja az átlagot és a szórást, majd seedenként részletes bontásban is kiírja az értékeket
'''
def print_multi_seed_summary(results):
    print("\n=== MULTI-SEED SUMMARY ===")

    metrics = [
        "bob_final_acc",
        "eve_final_acc",
        "bob_wrong_key_final_acc",
        "bob_wrong_nonce_final_acc",
        "naive_cipher_final_acc",
        "bob_final_hard_err",
        "eve_final_hard_err",
    ]

    for metric in metrics:
        values = [r[metric] for r in results]
        m, s = mean_std(values)
        print(f"{metric}: {m:.6f} ± {s:.6f}")

    print("\n=== PER-SEED RESULTS ===")
    for r in results:
        print(
            f"seed={r['seed']} | "
            f"Bob={r['bob_final_acc']:.4f} | "
            f"Eve={r['eve_final_acc']:.4f} | "
            f"WrongKey={r['bob_wrong_key_final_acc']:.4f} | "
            f"WrongNonce={r['bob_wrong_nonce_final_acc']:.4f} | "
            f"Naive={r['naive_cipher_final_acc']:.4f}"
        )

'''
A több seed-en futtatott kísérletek eredményeinek részletes statisztikai jellemzését adja meg
'''
def print_distribution_summary(results):
    print("\n=== DISTRIBUTION SUMMARY ===")

    metrics = [
        "bob_final_acc",
        "eve_final_acc",
        "bob_wrong_key_final_acc",
        "bob_wrong_nonce_final_acc",
        "naive_cipher_final_acc",
        "bob_final_hard_err",
        "eve_final_hard_err",
    ]

    for metric in metrics:
        values = [r[metric] for r in results]
        stats = descriptive_stats(values)

        print(f"\n[{metric}]")
        print(f"mean:   {stats['mean']:.6f}")
        print(f"std:    {stats['std']:.6f}")
        print(f"min:    {stats['min']:.6f}")
        print(f"q25:    {stats['q25']:.6f}")
        print(f"median: {stats['median']:.6f}")
        print(f"q75:    {stats['q75']:.6f}")
        print(f"max:    {stats['max']:.6f}")

'''
Hisztogramok segítségével vizualizálja a különböző metrikák eloszlását a több seed-en futtatott kísérletek során
'''
def plot_metric_distributions(results):
    metrics = [
        ("bob_final_acc", "Bob final accuracy"),
        ("eve_final_acc", "Eve final accuracy"),
        ("bob_wrong_key_final_acc", "Bob wrong-key final accuracy"),
        ("bob_wrong_nonce_final_acc", "Bob wrong-nonce final accuracy"),
        ("naive_cipher_final_acc", "Naive cipher final accuracy"),
        ("bob_final_hard_err", "Bob final hard error"),
        ("eve_final_hard_err", "Eve final hard error"),
    ]

    for metric_key, metric_name in metrics:
        values = [r[metric_key] for r in results]

        plt.figure(figsize=(8, 4))
        plt.hist(values, bins=min(10, len(values)))
        plt.title(f"Distribution of {metric_name}")
        plt.xlabel(metric_name)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

'''
Boxplot diagramokon ábrázolja a különböző metrikák eloszlását több seed-en futtatott kísérletek alapján
'''
def plot_metric_boxplots(results):
    metrics = [
        ("bob_final_acc", "Bob final acc"),
        ("eve_final_acc", "Eve final acc"),
        ("bob_wrong_key_final_acc", "Bob wrong-key acc"),
        ("bob_wrong_nonce_final_acc", "Bob wrong-nonce acc"),
        ("naive_cipher_final_acc", "Naive cipher acc"),
        ("bob_final_hard_err", "Bob hard err"),
        ("eve_final_hard_err", "Eve hard err"),
    ]

    data = [[r[key] for r in results] for key, _ in metrics]
    labels = [label for _, label in metrics]

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, tick_labels=labels)
    plt.title("Metric distributions across seeds")
    plt.ylabel("Value")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

# ============================================================
# TRAIN LOOP
# ============================================================

def train_game(
    iters=100_000,
    batch_size=256,
    lr_alice=2e-5,
    lr_bob=1e-3,
    lr_eve=1e-4,
    lambda_eve=LAMBDA_EVE,
    lambda_nonce=LAMBDA_NONCE,
    lambda_nonce_alice = LAMBDA_NONCE_ALICE,
    print_every=10000,
    eval_every=500,
    warmup=10_000,
    ramp=7_000,
    seed=42,
    use_eve_loss=True,
    use_ramp=True,
    use_nonce_loss=True,
    use_nonce_loss_alice = True,
):
    set_all_seeds(seed)
    alice = Alice().to(DEVICE)
    bob = Bob().to(DEVICE)
    eve = Eve().to(DEVICE)

    opt_a = optim.Adam(alice.parameters(), lr=lr_alice)
    opt_b = optim.Adam(bob.parameters(), lr=lr_bob)
    opt_e = optim.AdamW(eve.parameters(), lr=lr_eve, weight_decay=5e-4)

    history = {
        "iter": [],
        "bob_acc": [],
        "eve_acc": [],
        "bob_wrong_key_acc": [],
        "bob_wrong_nonce_acc": [],
        "naive_cipher_acc": [],
        "loss_bob": [],
        "loss_bob_wrong_nonce_term": [],
        "loss_eve": [],
        "loss_alice": [],
        "bob_hard_err": [],
        "eve_hard_err": [],
        "use_eve_loss": use_eve_loss,
        "use_ramp": use_ramp,
        "use_nonce_loss": use_nonce_loss,
    }

    for it in range(1, iters + 1):
        msg, key, nonce = generate_batch(batch_size)

        '''
        Ellenőrizzük a use_ramp változó értékét, így szabályozni tudjuk Eve kezdeti hatását (és befolyásolja a warmup változó értékét)
        - Ha True, akkor három fázist hoz létre:
                1. Warmup fázis:
                    sched = 0
                    lambda_now = 0
                    Eve-nek nincs hatása Alice-re
                2. Ramp fázis:
                    sched lineárisan nő (0 -> 1)
                    Eve egyre nagyobb tényező lesz Alice számára
                3. Stabil fázis:
                    sched = 1
                    lambda_now = lambda_eve
                    Eve teljes hatással lesz Alice működésére
        - Ha False, akkor Eve a kezdetektől teljes hatással lesz Alice működésére
        '''
        if use_ramp:
            sched = 0.0 if it < warmup else min(1.0, (it - warmup) / ramp)
            lambda_now = lambda_eve * sched
        else:
            warmup = 0
            lambda_now = lambda_eve

        # ----------------------------------------------------
        # Train Eve
        # ----------------------------------------------------
        '''
        Eve globális iteráción belüli frissüléseinek számának meghatározása
        A warmup változó értéke határozza meg, hogy Eve hanyadik iteráció után, mekkora intenzitással kezd el tanulni
            - warmup előtt:
                Eve nem tanul
            - warmup után, de warmup + 5000 iteráció előtt:
                Eve lassan elkezd tanulni, de még nem teljes intenzitással
            - warmup + 5000 után:
                Eve teljes intenzitással kezd el tanulni

        (Ezt az értéket a use_ramp változó értéke is befolyásolja) 
        '''
        eve_steps = 0 if it < warmup else (1 if it < warmup + 5000 else 2)

        for _ in range(eve_steps):
            # Alice és Bob befagyasztása
            set_requires_grad(alice, False)
            set_requires_grad(bob, False)
            set_requires_grad(eve, True)

            # Titkosított jel előállítása
            with torch.no_grad():
                cipher = alice_build_cipher(alice, msg, key, nonce)

            opt_e.zero_grad()
            # Eve becslést készít az üzenetre (a kulcs használata nélkül)
            eve_logits = eve(cipher, nonce)
            # A becslés helyességének vizsgálata
            loss_e = bce(eve_logits, msg)
            # Hibák visszaterjesztése Eve hálózatán
            loss_e.backward()
            # Eve paramétereinek frissítése
            opt_e.step()

        # ----------------------------------------------------
        # Train Bob
        # ----------------------------------------------------
        for _ in range(BOB_STEPS_PER_ITER):
            # Alice és Eve befagyasztása
            set_requires_grad(alice, False)
            set_requires_grad(bob, True)
            set_requires_grad(eve, False)

            # Titkosított jel előállítása
            with torch.no_grad():
                cipher = alice_build_cipher(alice, msg, key, nonce)

            opt_b.zero_grad()

            # Helyes nonce-szal történő dekódolás
            bob_logits = bob(cipher, key, nonce)
            loss_b_correct = bce(bob_logits, msg)

            # Rossz nonce-os kontroll
            wrong_nonce = torch.roll(nonce, shifts=1, dims=0)
            bob_wrong_nonce_logits = bob(cipher, key, wrong_nonce)

            bob_wrong_nonce_prob = torch.sigmoid(bob_wrong_nonce_logits)
            bob_wrong_nonce_soft_bit_error = torch.abs(msg - bob_wrong_nonce_prob).sum(dim=1).mean()

            n_half = MSG_BITS / 2.0
            bob_wrong_nonce_random_term = ((n_half - bob_wrong_nonce_soft_bit_error) ** 2) / (n_half ** 2)

            loss_b = loss_b_correct
            if use_nonce_loss:
                loss_b = loss_b + lambda_nonce * bob_wrong_nonce_random_term

            loss_b.backward()
            opt_b.step()

        # ----------------------------------------------------
        # Train Alice
        # ----------------------------------------------------
        for _ in range(ALICE_STEPS_PER_ITER):
            # Bob és Eve befagyasztása
            set_requires_grad(alice, True)
            set_requires_grad(bob, False)
            set_requires_grad(eve, False)

            opt_a.zero_grad()

            # Cipher generálása
            cipher = alice_build_cipher(alice, msg, key, nonce)
            # Bob próbálkozása dekódolni a ciphert a kulcs segítségével
            bob_logits = bob(cipher, key, nonce)
            # Eve próbálkozása dekódolni a ciphert a kulcs segítsége nélkül
            eve_logits = eve(cipher, nonce)

            # Bob-loss kiszámítása
            loss_bob_for_alice = bce(bob_logits, msg)

            wrong_nonce = torch.roll(nonce, shifts=1, dims=0)
            bob_wrong_nonce_logits = bob(cipher, key, wrong_nonce)

            bob_wrong_nonce_prob = torch.sigmoid(bob_wrong_nonce_logits)
            bob_wrong_nonce_soft_bit_error = torch.abs(msg - bob_wrong_nonce_prob).sum(dim=1).mean()

            n_half = MSG_BITS / 2.0
            bob_wrong_nonce_random_term = ((n_half - bob_wrong_nonce_soft_bit_error) ** 2) / (n_half ** 2)

            eve_prob = torch.sigmoid(eve_logits)
            eve_soft_bit_error = torch.abs(msg - eve_prob).sum(dim=1).mean()

            n_half = MSG_BITS / 2.0
            eve_random_term = ((n_half - eve_soft_bit_error) ** 2) / (n_half ** 2)

            loss_a = loss_bob_for_alice

            '''
            Alice-loss összerakása
                Ha use_eve_loss = False és use_nonce_loss_alice = False:
                    - Alice csak a Bob-loss-t veszi figyelembe
                Ha use_eve_loss = True:
                    - loss_a = loss_bob_for_alice + lambda_now * eve_random_term
                    (Alice próbálja rontani Eve teljesítményét)
                Ha use_nonce_loss_alice = True:
                    - loss_a = loss_bob_for_alice + lambda_nonce_alice * bob_wrong_nonce_random_term
                    (Alice próbálja elérni, hogy rossz nonce esetén Bob rosszul dekódoljon)
                Ha mindkettő True:
                    - loss_a = loss_bob_for_alice + lambda_now * eve_random_term + lambda_nonce_alice * bob_wrong_nonce_random_term
            '''
            if use_eve_loss:
                if it >= warmup or not use_ramp:
                    loss_a = loss_a + lambda_now * eve_random_term

            if use_nonce_loss_alice:
                loss_a = loss_a + lambda_nonce_alice * bob_wrong_nonce_random_term

            # Hibák visszaterjesztése Alice hálózatán
            loss_a.backward()
            # Alice paramétereinek frissítése
            opt_a.step()

        # -------------------------------------------------------
        # A MODELL KIÉRTÉKELÉSE ÉS A MÉRÉSI EREDMÉNYEK NAPLÓZÁSA
        # -------------------------------------------------------
        if it % eval_every == 0 or it == 1:
            with torch.no_grad():
                # Új tesztadatok létrehozása az aktuális állapot mérésére
                msg_t, key_t, nonce_t = generate_batch(batch_size)
                # A titkosított jel előállítása Alice által
                cipher_t = alice_build_cipher(alice, msg_t, key_t, nonce_t)

                # Bob visszafejtési kísérlete a kulcs használatával
                bob_logits_t = bob(cipher_t, key_t, nonce_t)
                # Eve visszafejtési kísérlete a kulcs használata nélkül
                eve_logits_t = eve(cipher_t, nonce_t)

                # Szándékosan rossz kulcsos változat létrehozása a rendszer kulcsfüggőségének vizsgálatához
                wrong_key_t = torch.roll(key_t, shifts=1, dims=0)
                # Bob visszafejtési kísérlete a rossz kulcs használatával
                bob_wrong_key_logits_t = bob(cipher_t, wrong_key_t, nonce_t)

                # Szándékosan rossz nonce-os változat létrehozása a rendszer nonce függőségének vizsgálatához
                wrong_nonce_t = torch.roll(nonce_t, shifts=1, dims=0)
                # Bob visszafejtési kísérlete a rossz nonce használatával
                bob_wrong_nonce_logits_t = bob(cipher_t, key_t, wrong_nonce_t)

                # Bitpontos pontosságok meghatározása
                bob_acc = bitwise_accuracy_from_logits(bob_logits_t, msg_t)
                eve_acc = bitwise_accuracy_from_logits(eve_logits_t, msg_t)
                bob_wrong_key_acc = bitwise_accuracy_from_logits(bob_wrong_key_logits_t, msg_t)
                bob_wrong_nonce_acc = bitwise_accuracy_from_logits(bob_wrong_nonce_logits_t, msg_t)

                # Bob és Eve veszteségértékei
                loss_b_eval = bce(bob_logits_t, msg_t).item()
                loss_e_eval = bce(eve_logits_t, msg_t).item()

                # Az átlagos hibás bitszám mintánkénti kiszámítása
                bob_hard_err = hard_bit_error_from_logits(bob_logits_t, msg_t)
                eve_hard_err = hard_bit_error_from_logits(eve_logits_t, msg_t)

                # Annak vizsgálata, hogy a cipher előjeléből mennyi információ olvasható ki,
                # így annak vizsgálata, hogy van-e nyers szivárgás
                naive_pred = (cipher_t > 0).float()
                naive_acc = (naive_pred == msg_t).float().mean().item()

                bob_wrong_nonce_prob_eval = torch.sigmoid(bob_wrong_nonce_logits_t)
                bob_wrong_nonce_soft_bit_error_eval = torch.abs(msg_t - bob_wrong_nonce_prob_eval).sum(dim=1).mean()
                bob_wrong_nonce_random_term_eval = ((n_half - bob_wrong_nonce_soft_bit_error_eval) ** 2) / (n_half ** 2)

            # Az értékek eltárolása későbbi feldolgozásra
            history["iter"].append(it)
            history["bob_acc"].append(bob_acc)
            history["eve_acc"].append(eve_acc)
            history["bob_wrong_key_acc"].append(bob_wrong_key_acc)
            history["bob_wrong_nonce_acc"].append(bob_wrong_nonce_acc)
            history["naive_cipher_acc"].append(naive_acc)
            history["loss_bob"].append(loss_b_eval)
            history["loss_bob_wrong_nonce_term"].append(bob_wrong_nonce_random_term_eval.item())
            history["loss_eve"].append(loss_e_eval)
            history["loss_alice"].append(loss_a.item())
            history["bob_hard_err"].append(bob_hard_err)
            history["eve_hard_err"].append(eve_hard_err)

        # Az aktuális mérési állapot kiíratása futás közben
        if it % print_every == 0 or it == 1:
            print(
                f"Iter {it:6d} | "
                f"Bob acc={history['bob_acc'][-1]:.4f} | "
                f"Eve acc={history['eve_acc'][-1]:.4f} | "
                f"Bob wrong-key acc={history['bob_wrong_key_acc'][-1]:.4f} | "
                f"Bob wrong-nonce acc={history['bob_wrong_nonce_acc'][-1]:.4f} | "
                f"Naive cipher acc={history['naive_cipher_acc'][-1]:.4f} | "
                f"L_bob={history['loss_bob'][-1]:.4f} | "
                f"L_bob_nonce={history['loss_bob_wrong_nonce_term'][-1]:.4f} | "
                f"L_eve={history['loss_eve'][-1]:.4f} | "
                f"L_alice={history['loss_alice'][-1]:.4f}"
            )

    return alice, bob, eve, history

# ============================================================
# PLOTS
# ============================================================

'''
A fő tréning közbeni viselkedés kirajzolása
'''
def plot_history(history):
    iters = history["iter"]

    plt.figure(figsize=(12, 5))
    plt.plot(iters, history["bob_acc"], label="Bob accuracy")
    plt.plot(iters, history["eve_acc"], label="Eve accuracy")
    plt.plot(iters, history["bob_wrong_key_acc"], label="Bob wrong-key accuracy")
    plt.plot(iters, history["bob_wrong_nonce_acc"], label="Bob wrong-nonce accuracy")
    plt.plot(iters, history["naive_cipher_acc"], label="Naive cipher accuracy")
    plt.title("Accuracy over training")
    plt.xlabel("Iteration")
    plt.ylabel("Bitwise accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(iters, history["loss_bob"], label="Bob BCE loss")
    plt.plot(iters, history["loss_bob_wrong_nonce_term"], label="Bob wrong-nonce term")
    plt.plot(iters, history["loss_eve"], label="Eve BCE loss")
    plt.plot(iters, history["loss_alice"], label="Alice objective")
    plt.title("Losses over training")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(iters, history["bob_hard_err"], label="Bob hard bit error")
    plt.plot(iters, history["eve_hard_err"], label="Eve hard bit error")
    plt.axhline(MSG_BITS / 2.0, linestyle="--", label="Random level")
    plt.title("Hard bit error over training")
    plt.xlabel("Iteration")
    plt.ylabel("Avg wrong bits / sample")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# GYORSTESZT
# ============================================================

'''
A modell működésének szemléltetése egyetlen mintán

Egy véletlen üzenet, kulcs és nonce generálása után, előállítja a titkosított reprezentációt (cipher),
 majd kiírja annak nyers és előjeles (binárisított) formáját, valamint Bob és Eve dekódolt kimenetét
'''
@torch.no_grad()
def run_quick_demo(alice, bob, eve):
    msg, key, nonce = generate_batch(batch_size=1)
    cipher = alice_build_cipher(alice, msg, key, nonce)

    bob_out = (torch.sigmoid(bob(cipher, key, nonce)) > 0.5).int()
    eve_out = (torch.sigmoid(eve(cipher, nonce)) > 0.5).int()

    print("\nMESSAGE:     ", msg.int().cpu().numpy())
    print("KEY:         ", key.int().cpu().numpy())
    print("NONCE:       ", nonce.int().cpu().numpy())
    print("CIPHER RAW:  ", cipher.cpu().numpy())
    print("CIPHER SIGN: ", (cipher > 0).int().cpu().numpy())
    print("BOB OUT:     ", bob_out.cpu().numpy())
    print("EVE OUT:     ", eve_out.cpu().numpy())

'''
A nonce hatásának szemléltetése fix üzenet és kulcs mellett.

Egy rögzített üzenet és kulcs generálása után több különböző nonce használata:
    - minden nonce esetén új cipher jön létre,
    - majd kiírja a cipher-t és annak előjeles változatát,
    - valamint Bob és Eve dekódolt kimenetét.

Cél:
Megmutatni, hogy ugyanaz az üzenet és kulcs különböző nonce-ok mellett
eltérő titkosított reprezentációkhoz vezet, miközben Bob továbbra is
képes dekódolni, Eve viszont bizonytalanabb marad.
'''
@torch.no_grad()
def run_nonce_demo(alice, bob, eve, num_trials=3):
    msg = torch.randint(0, 2, (1, MSG_BITS), device=DEVICE).float()
    key = torch.randint(0, 2, (1, KEY_BITS), device=DEVICE).float()

    print("\nFIXED MESSAGE:", msg.int().cpu().numpy())
    print("FIXED KEY:    ", key.int().cpu().numpy())

    for i in range(num_trials):
        nonce = torch.randint(0, 2, (1, NONCE_BITS), device=DEVICE).float()
        cipher = alice_build_cipher(alice, msg, key, nonce)

        bob_out = (torch.sigmoid(bob(cipher, key, nonce)) > 0.5).int()
        eve_out = (torch.sigmoid(eve(cipher, nonce)) > 0.5).int()

        print(f"\n--- TRIAL {i + 1} ---")
        print("NONCE:       ", nonce.int().cpu().numpy())
        print("CIPHER RAW:  ", cipher.cpu().numpy())
        print("CIPHER SIGN: ", (cipher > 0).int().cpu().numpy())
        print("BOB OUT:     ", bob_out.cpu().numpy())
        print("EVE OUT:     ", eve_out.cpu().numpy())

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Device:", DEVICE)

    SEEDS = [11, 22]
    # SEEDS = [11, 22, 33, 44, 55]
    '''
    SEEDS = [
        11, 22, 33, 44, 55,
        66, 77, 88, 99, 111,
        122, 133, 144, 155, 166,
        177, 188, 199, 211, 222,
        233, 244, 255, 266, 277
    ]
    '''

    all_results = []
    last_run = None

    '''
    A kísérlet végrehajtása
    Elindul a tanítási folyamat, eltárolja az adott futási folyamat eredményeit
    '''
    for seed in SEEDS:
        print("\n" + "=" * 60)
        print(f"RUN START | seed = {seed}")
        print("=" * 60)

        alice, bob, eve, hist = train_game(
            iters=100_000,
            batch_size=256,
            lr_alice=2e-5,
            lr_bob=1e-3,
            lr_eve=1e-4,
            lambda_eve=LAMBDA_EVE,
            lambda_nonce=LAMBDA_NONCE,
            lambda_nonce_alice= LAMBDA_NONCE_ALICE,
            print_every=10000,
            eval_every=500,
            warmup=10_000,
            ramp=7_000,
            seed=seed,
            use_eve_loss=USE_EVE_LOSS,
            use_ramp=USE_RAMP,
            use_nonce_loss=USE_NONCE_LOSS,
            use_nonce_loss_alice= USE_NONCE_LOSS_ALICE,
        )

        # Eredmények összegyűjtése
        run_result = {
            "seed": seed,
            "bob_final_acc": hist["bob_acc"][-1],
            "eve_final_acc": hist["eve_acc"][-1],
            "bob_wrong_key_final_acc": hist["bob_wrong_key_acc"][-1],
            "bob_wrong_nonce_final_acc": hist["bob_wrong_nonce_acc"][-1],
            "naive_cipher_final_acc": hist["naive_cipher_acc"][-1],
            "bob_final_hard_err": hist["bob_hard_err"][-1],
            "eve_final_hard_err": hist["eve_hard_err"][-1],
        }

        all_results.append(run_result)
        last_run = (alice, bob, eve, hist)

        print("\n--- RUN RESULT ---")
        print(f"seed:                    {seed}")
        print(f"bob_final_acc:           {run_result['bob_final_acc']:.6f}")
        print(f"eve_final_acc:           {run_result['eve_final_acc']:.6f}")
        print(f"bob_wrong_key_acc:       {run_result['bob_wrong_key_final_acc']:.6f}")
        print(f"bob_wrong_nonce_acc:     {run_result['bob_wrong_nonce_final_acc']:.6f}")
        print(f"naive_cipher_acc:        {run_result['naive_cipher_final_acc']:.6f}")
        print(f"bob_final_hard_err:      {run_result['bob_final_hard_err']:.6f}")
        print(f"eve_final_hard_err:      {run_result['eve_final_hard_err']:.6f}")

    # Opcionálisan az utolsó futás külön vizsgálata
    if last_run is not None:
        alice, bob, eve, hist = last_run
        plot_history(hist)
        run_quick_demo(alice, bob, eve)
        run_nonce_demo(alice, bob, eve, num_trials=3)

    # Összesített kiértékelés
    print_multi_seed_summary(all_results)
    print_distribution_summary(all_results)
    plot_metric_distributions(all_results)
    plot_metric_boxplots(all_results)
    # save_multi_seed_results(all_results)