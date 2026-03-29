# szakdolgozat
*TESZT PROGRAM*

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

A modell tartalmaz 3 darab opcionálisan kikapcsolható komponenst is, melyek az Alice-loss kiszámításában játszanak szerepet:
- Eve-loss komponens,
- mask regularizáció,
- ramp/warmup mechanizmus Eve fokozatos bevezetésére.

A program első sorban több seed melletti futtatást támogat, és a végén statisztikai összesítést,
eloszlásvizsgálatot, valamint grafikus megjelenítést készít az eredményekről.
