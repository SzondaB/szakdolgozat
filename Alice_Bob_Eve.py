"""
Ez a program egy Alice-Bob-Eve típusú neurális kommunikációs modellt valósít meg.

Alice egy bináris üzenet és egy kulcs alapján titkosított reprezentációt állít elő.
Bob a titkosított reprezentáció és a helyes kulcs segítségével próbálja rekonstruálni az üzenetet,
míg Eve ugyanezt a feladatot végzi kulcs nélkül.

A modell elsődleges célja annak vizsgálata, hogy a neurális hálózatok mennyire képesek
alkalmazkodni ismeretlen és változó jelsorozatokhoz, illetve hogy a dekódoló hálózat
mennyire marad működőképes a jelstruktúra változásai mellett.

A tréning három elkülönített részből áll:
1. Eve tanítása a cipherből történő kulcs nélküli visszafejtésre,
2. Bob tanítása a cipher és a helyes kulcs alapján történő dekódolásra,
3. Alice tanítása olyan reprezentáció előállítására, amely Bob számára jól dekódolható,
   Eve számára viszont nehezebben értelmezhető.

A modell opcionálisan tartalmaz:
- Eve-loss komponenst,
- mask regularizációt,
- ramp/warmup mechanizmust Eve fokozatos bevezetésére.

A program több seed melletti futtatást támogat, és a végén statisztikai összesítést,
eloszlásvizsgálatot, valamint grafikus megjelenítést készít az eredményekről.
"""

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

MSG_BITS = 32 # az üzenet hosszúsága
KEY_BITS = 32 # a kulcs hosszúsága

LAMBDA_EVE = 1.0 # mérőszám, Alice Eve randomizálására való törekvéséhez
LAMBDA_MASK = 0.1 # Alice loss-jára mekkora hatással legyen a maszk
MASK_TARGET = 0.5 # átlagosan mennyire használja Alice az encoding branch-et

# attribútumok különböző tesztekhez
USE_EVE_LOSS = True
USE_MASK_LOSS = True
USE_RAMP = True

# globális iteráción belüli frissülések számának meghatározása
ALICE_STEPS_PER_ITER = 1
BOB_STEPS_PER_ITER = 2

# ============================================================
# ADATGENERÁLÁS
# ============================================================

def generate_batch(batch_size=128, device=DEVICE):
    msg = torch.randint(0, 2, (batch_size, MSG_BITS), device=device).float()
    key = torch.randint(0, 2, (batch_size, KEY_BITS), device=device).float()
    return msg, key

# ============================================================
# MODELLEK (ALICE, BOB, EVE)
# ============================================================

'''
Alice
Bemenet: random generált üzenet és kulcs
Kimenet: egy kódolt reprezentáció és egy maszk
'''
class Alice(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(MSG_BITS + KEY_BITS, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.enc_head = nn.Linear(128, MSG_BITS)
        self.mask_head = nn.Linear(128, MSG_BITS)

    def forward(self, msg, key):
        x = torch.cat([msg, key], dim=1)
        h = self.backbone(x)
        enc_logits = self.enc_head(h)
        mask_logits = self.mask_head(h)
        return enc_logits, mask_logits

'''
Bob
Bemenet: titkosított üzenet és kulcs
Kimenet: az eredeti üzenet becsült bitje (csak logitok)
'''
class Bob(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MSG_BITS + KEY_BITS, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, MSG_BITS)
        )

    def forward(self, cipher, key):
        x = torch.cat([cipher, key], dim=1)
        return self.net(x)

"""
Eve
Bemenet: csak a titkosított üzenet
Kimenet: az eredeti üzenet becsült bitje (csak logitok)
"""
class Eve(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MSG_BITS, 64),
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

    def forward(self, cipher):
        return self.net(cipher)

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
Alice kimenetéből előállítja a tényleges titkosított üzenetet és a hozzá tartozó maszkot
Az encoding ág kimenetét hiperbolikus tangens aktivációval korlátozza, míg a maszk sigmoid aktivációval [0,1] tartományba kerül
A végső cipher a maszk és az encoding elemenkénti szorzataként jön létre
'''
def alice_build_cipher(alice, msg, key):
    enc_logits, mask_logits = alice(msg, key)

    enc = torch.tanh(enc_logits)
    mask = torch.sigmoid(mask_logits)

    cipher = mask * enc
    return cipher, mask

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
        "naive_cipher_final_acc",
        "bob_final_hard_err",
        "eve_final_hard_err",
        "mask_mean_final",
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
            f"Naive={r['naive_cipher_final_acc']:.4f} | "
            f"Mask={r['mask_mean_final']:.4f}"
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
        "naive_cipher_final_acc",
        "bob_final_hard_err",
        "eve_final_hard_err",
        "mask_mean_final",
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
        ("naive_cipher_final_acc", "Naive cipher final accuracy"),
        ("bob_final_hard_err", "Bob final hard error"),
        ("eve_final_hard_err", "Eve final hard error"),
        ("mask_mean_final", "Mask mean final"),
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
        ("naive_cipher_final_acc", "Naive cipher acc"),
        ("bob_final_hard_err", "Bob hard err"),
        ("eve_final_hard_err", "Eve hard err"),
        ("mask_mean_final", "Mask mean"),
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
    lambda_mask=LAMBDA_MASK,
    mask_target=MASK_TARGET,
    print_every=10000,
    eval_every=500,
    warmup=10_000,
    ramp=7_000,
    seed = 42,
    use_eve_loss=True,
    use_mask_loss=True,
    use_ramp=True,
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
        "naive_cipher_acc": [],
        "loss_bob": [],
        "loss_eve": [],
        "loss_alice": [],
        "bob_hard_err": [],
        "eve_hard_err": [],
        "mask_mean": [],
        "use_eve_loss": use_eve_loss,
        "use_mask_loss": use_mask_loss,
        "use_ramp": use_ramp,
    }

    for it in range(1, iters + 1):
        msg, key = generate_batch(batch_size)

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
                cipher, _ = alice_build_cipher(alice, msg, key)

            opt_e.zero_grad()
            # Eve becslést készít az üzenetre (a kulcs használata nélkül)
            eve_logits = eve(cipher)
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
                cipher, _ = alice_build_cipher(alice, msg, key)

            opt_b.zero_grad()
            # Bob becslést készít az üzenetre (a kulcs használatával)
            bob_logits = bob(cipher, key)
            # A becslés helyességének vizsgálata
            loss_b = bce(bob_logits, msg)
            # Hibák visszaterjesztése Bob hálózatán
            loss_b.backward()
            # Bob paramétereinek frissítése
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
            cipher, mask = alice_build_cipher(alice, msg, key)
            # Bob próbálkozása dekódolni a ciphert a kulcs segítségével
            bob_logits = bob(cipher, key)
            # Eve próbálkozása dekódolni a ciphert a kulcs segítsége nélkül
            eve_logits = eve(cipher)

            # Bob-loss kiszámítása
            loss_bob_for_alice = bce(bob_logits, msg)


            eve_prob = torch.sigmoid(eve_logits)
            eve_soft_bit_error = torch.abs(msg - eve_prob).sum(dim=1).mean()

            n_half = MSG_BITS / 2.0
            eve_random_term = ((n_half - eve_soft_bit_error) ** 2) / (n_half ** 2)

            # Mask regularizáció kiszámítása (megakadályozza a full encoding vagy a null encoding előfordulását)
            mask_mean_batch = mask.mean()
            mask_reg = (mask_mean_batch - mask_target) ** 2

            '''
            Alice-loss összerakása
                Ha use_eve_loss és use_mask_loss változók értéke False:
                    - Alice csak a Bob-loss-t veszi figyelembe
                Ha use_eve_loss változó True és use_mask_loss változó False:
                    - loss_a = loss_bob_for_alice + lambda_now * eve_random_term
                Ha use_eve_loss és use_mask_loss változók értéke True:
                    - loss_a = loss_bob_for_alice + lambda_now * eve_random_term + lambda_mask * mask_reg
            '''
            loss_a = loss_bob_for_alice

            if use_eve_loss:
                if it >= warmup or not use_ramp:
                    loss_a = loss_a + lambda_now * eve_random_term

            if use_mask_loss:
                loss_a = loss_a + lambda_mask * mask_reg

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
                msg_t, key_t = generate_batch(batch_size)
                # A titkosított jel és a hozzá tartozó masz előállítása Alice által
                cipher_t, mask_t = alice_build_cipher(alice, msg_t, key_t)

                # Bob visszafejtési kísérlete a kulcs használatával
                bob_logits_t = bob(cipher_t, key_t)
                # Eve visszafejtési kísérlete a kulcs használata nélkül
                eve_logits_t = eve(cipher_t)

                # Szándékosan rossz kulcsos változat létrehozása a rendszer kulcsfüggőségének vizsgálatához
                wrong_key_t = torch.roll(key_t, shifts=1, dims=0)
                # Bob visszafejtési kísérlete a rossz kulcs használatával
                bob_wrong_key_logits_t = bob(cipher_t, wrong_key_t)

                # Bitpontos pontosságok meghatározása
                bob_acc = bitwise_accuracy_from_logits(bob_logits_t, msg_t)
                eve_acc = bitwise_accuracy_from_logits(eve_logits_t, msg_t)
                bob_wrong_key_acc = bitwise_accuracy_from_logits(bob_wrong_key_logits_t, msg_t)

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

                # A maszk átlagértékének kiszámítása, ami megmutatja, hogy Alice mennyire használja az encoding ágat
                mask_mean_eval = mask_t.mean().item()


            # Az értékek eltárolása későbbi feldolgozásra
            history["iter"].append(it)
            history["bob_acc"].append(bob_acc)
            history["eve_acc"].append(eve_acc)
            history["bob_wrong_key_acc"].append(bob_wrong_key_acc)
            history["naive_cipher_acc"].append(naive_acc)
            history["loss_bob"].append(loss_b_eval)
            history["loss_eve"].append(loss_e_eval)
            history["loss_alice"].append(loss_a.item())
            history["bob_hard_err"].append(bob_hard_err)
            history["eve_hard_err"].append(eve_hard_err)
            history["mask_mean"].append(mask_mean_eval)

        # Az aktuális mérési állapot kiíratása futás közben
        if it % print_every == 0 or it == 1:
            print(
                f"Iter {it:6d} | "
                f"Bob acc={history['bob_acc'][-1]:.4f} | "
                f"Eve acc={history['eve_acc'][-1]:.4f} | "
                f"Bob wrong-key acc={history['bob_wrong_key_acc'][-1]:.4f} | "
                f"Naive cipher acc={history['naive_cipher_acc'][-1]:.4f} | "
                f"Mask mean={history['mask_mean'][-1]:.4f} | "
                f"L_bob={history['loss_bob'][-1]:.4f} | "
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
    plt.plot(iters, history["naive_cipher_acc"], label="Naive cipher accuracy")
    plt.title("Accuracy over training")
    plt.xlabel("Iteration")
    plt.ylabel("Bitwise accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(iters, history["loss_bob"], label="Bob BCE loss")
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

    plt.figure(figsize=(12, 5))
    plt.plot(iters, history["mask_mean"], label="Mask mean")
    plt.axhline(MASK_TARGET, linestyle="--", label="Mask target")
    plt.title("Mask mean over training")
    plt.xlabel("Iteration")
    plt.ylabel("Mean mask value")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# GYORSTESZT
# ============================================================

'''
A modell működésének egyetlen mintán való szemléltetése
Egy véletlen üzenet és kulcs generálása után megjeleníti a titkosított jelet,
a maszk értékeit, valamint Bob és Eve dekódolt kimenetét
'''
@torch.no_grad()
def run_quick_demo(alice, bob, eve):
    msg, key = generate_batch(batch_size=1)
    cipher, mask = alice_build_cipher(alice, msg, key)

    bob_out = (torch.sigmoid(bob(cipher, key)) > 0.5).int()
    eve_out = (torch.sigmoid(eve(cipher)) > 0.5).int()

    print("\nMESSAGE:     ", msg.int().cpu().numpy())
    print("KEY:         ", key.int().cpu().numpy())
    print("MASK:        ", mask.cpu().numpy())
    print("MASK>0.5:    ", (mask > 0.5).int().cpu().numpy())
    print("CIPHER RAW:  ", cipher.cpu().numpy())
    print("CIPHER SIGN: ", (cipher > 0).int().cpu().numpy())
    print("BOB OUT:     ", bob_out.cpu().numpy())
    print("EVE OUT:     ", eve_out.cpu().numpy())

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Device:", DEVICE)

    # A seed lista megadása
    SEEDS = [11, 22, 33, 44, 55]
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
            lambda_eve=1.0,
            lambda_mask=LAMBDA_MASK,
            mask_target=MASK_TARGET,
            print_every=10000,
            eval_every=500,
            warmup=10_000,
            ramp=7_000,
            seed=seed,
            use_eve_loss=USE_EVE_LOSS,
            use_mask_loss=USE_MASK_LOSS,
            use_ramp=USE_RAMP,
        )

        # Eredmények összegyűjtése
        run_result = {
            "seed": seed,
            "bob_final_acc": hist["bob_acc"][-1],
            "eve_final_acc": hist["eve_acc"][-1],
            "bob_wrong_key_final_acc": hist["bob_wrong_key_acc"][-1],
            "naive_cipher_final_acc": hist["naive_cipher_acc"][-1],
            "bob_final_hard_err": hist["bob_hard_err"][-1],
            "eve_final_hard_err": hist["eve_hard_err"][-1],
            "mask_mean_final": hist["mask_mean"][-1],
        }

        all_results.append(run_result)
        last_run = (alice, bob, eve, hist)

        print("\n--- RUN RESULT ---")
        print(f"seed:                 {seed}")
        print(f"bob_final_acc:        {run_result['bob_final_acc']:.6f}")
        print(f"eve_final_acc:        {run_result['eve_final_acc']:.6f}")
        print(f"bob_wrong_key_acc:    {run_result['bob_wrong_key_final_acc']:.6f}")
        print(f"naive_cipher_acc:     {run_result['naive_cipher_final_acc']:.6f}")
        print(f"bob_final_hard_err:   {run_result['bob_final_hard_err']:.6f}")
        print(f"eve_final_hard_err:   {run_result['eve_final_hard_err']:.6f}")
        print(f"mask_mean_final:      {run_result['mask_mean_final']:.6f}")

    # Opcionálisan az utolsó futás külön vizsgálata
    if last_run is not None:
        alice, bob, eve, hist = last_run

        plot_history(hist)
        run_quick_demo(alice, bob, eve)

    # Összesített kiértékelés
    print_multi_seed_summary(all_results)
    print_distribution_summary(all_results)
    plot_metric_distributions(all_results)
    plot_metric_boxplots(all_results)
    save_multi_seed_results(all_results)