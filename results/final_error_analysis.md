# Final Error Analysis: Single LLM vs Multi-Agent (Run 0)

**Generated:** 2026-04-07 | **Pairs:** 1916 | **Single LLM cached:** 1916 | **Multi-Agent cached:** 1916

## 1. Cohen's Kappa: Single LLM vs Multi-Agent

- **Pairs compared:** 1916
- **Agreement count:** 1891 (98.7%)
- **Cohen's Kappa:** 0.9369

### Agreement Matrix

|                          | Multi-Agent: NO MATCH | Multi-Agent: MATCH |
| ------------------------ | --------------------- | ------------------ |
| **Single LLM: NO MATCH** | 1679                  | 6                  |
| **Single LLM: MATCH**    | 19                    | 212                |

## 2. Per-Pipeline Error Summary

### Single LLM

| Metric    | Count    |
| --------- | -------- |
| TP        | 198      |
| FP        | 33       |
| FN        | 8        |
| TN        | 1677     |
| **Total** | **1916** |
| Precision | 0.8571   |
| Recall    | 0.9612   |
| F1        | 0.9062   |

#### False Positives (33)

| #   | Source ID | Source Name                                                     | Target ID | Target Name                                                     | Src Price | Tgt Price | Confidence | Reasoning                                                                                                                                                                                                   |
| --- | --------- | --------------------------------------------------------------- | --------- | --------------------------------------------------------------- | --------- | --------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | 1030      | transcend 8gb micro secure digital memory card ts8gusdhc6       | 333       | transcend 8gb secure digital high capacity ( sdhc ) card cla... | $39.00    | $11.45    | 0.75       | Both listings reference the same brand, capacity (8GB), class (6) and very similar model numbers (ts8gsdhc6 vs ts8gusdhc6, likely a typo). Descriptions overlap on being a microSDHC card. The price dif... |
| 2   | 1060      | canon eos rebel xsi silver digital slr camera xsireb1855s       | 662       | canon eos rebel xsi digital slr camera with ef-s 18-55mm f/3... | $799.00   | $649.88   | 0.85       | Both listings describe the Canon EOS Rebel XSi digital SLR with the same model number (2756B003) and identical core specs (12 MP, 18‑55mm lens). The only difference is color and price, which are typic... |
| 3   | 110       | canon photo ink cartridge cl52                                  | 84        | canon ink cartridge for pixma ip1600 , ip6210d and ip6220d p... | $25.00    | $19.11    | 0.6        | Both listings refer to a Canon CL‑52 ink cartridge compatible with the same printer models (IP6210D and IP6220D) and the part number 0617B002 matches the CL‑52 series; price difference is modest, sugg... |
| 4   | 175       | tech craft avalon series tv stand swp48                         | 279       | abs48 wood tv stand ( avalon series , 48-in . max )             | $299.00   | N/A       | 0.85       | Both listings reference an Avalon Series TV stand with a 48-inch width; the abbreviations (SWP48, ABS48) align, indicating they are the same model despite missing description for Product B.               |
| 5   | 257       | sony progressive scan silver dvd player dvpns57ps               | 351       | sony dvpns57p dvd player dvpns57pb                              | N/A       | $39.00    | 0.85       | Both listings reference Sony DVP-NS57P progressive scan DVD player with identical model base; suffixes differ only by color (silver vs black), indicating variants of the same product.                     |
| 6   | 386       | lg dlex8377nm navy blue xl capacity electric steamdryer dlex... | 453       | lg 27 ' front-load electric dryer with 7.3 cu . ft. capacity    | N/A       | N/A       | 0.65       | Both listings describe an LG front‑load electric dryer with 7.3 cu ft capacity; product A provides the specific model DLEX8377NM, which matches the generic description of product B, suggesting they re... |
| 7   | 444       | sony silver cyber-shot 7.2 megapixel digital camera dscw120     | 573       | sony cyber-shot dsc-w120 digital camera pink dscw120/p          | N/A       | $109.99   | 0.95       | Both listings reference the Sony DSC-W120 Cyber-shot camera model; despite different colors (silver vs pink) and slight description variations, they share the same model number and core specifications... |
| 8   | 486       | canon silver 8.0 megapixel powershot digital camera sd1100is    | 543       | canon powershot sd1100 is digital elph camera melody pink 25... | N/A       | N/A       | 0.85       | Both listings reference the Canon PowerShot SD1100 IS model with identical specs (8 MP, 3× optical zoom). The only difference is color (silver vs pink), which is a variant of the same product.            |
| 9   | 486       | canon silver 8.0 megapixel powershot digital camera sd1100is    | 544       | canon powershot sd1100 is digital elph camera rhythm & blue ... | N/A       | N/A       | 0.95       | Both listings reference the Canon PowerShot SD1100 IS model with 8 MP, 3× optical zoom and identical specifications; only color/packaging differs, indicating the same product.                             |
| 10  | 487       | canon blue 8.0 megapixel powershot digital camera sd1100isb     | 542       | canon powershot sd1100 is digital elph camera swing silver 2... | N/A       | N/A       | 0.93       | Both listings describe the Canon PowerShot SD1100 IS 8 MP camera with 3× optical zoom; the only difference is the color (blue vs silver), which is a variant of the same model.                             |
| 11  | 504       | panasonic viera 46 ' plasma flat panel 1080p hdtv in black t... | 843       | panasonic viera th-46pz850u 46 ' plasma tv th46pz850u           | N/A       | $1695.96  | 0.93       | Both listings describe a 46" Panasonic Viera plasma HDTV with the same model family (TH-46PZ850U), matching specifications (1080p, ATSC/NTSC, 1920x1080). The slight difference in the model string appe... |
| 12  | 520       | nikon coolpix s550 10 megapixel black digital camera coolpix... | 538       | nikon coolpix s550 digital camera cool blue 26109               | N/A       | $169.95   | 0.78       | Both listings identify the Nikon Coolpix S550 model with 10 MP sensor and 2.5" LCD; differences in color (black vs cool blue) and minor spec wording are typical variations, indicating the same product... |
| 13  | 529       | sony dvp-fx820 blue 8 ' portable dvd player dvpfx820li          | 763       | sony dvp-fx820 / r portable dvd player dvpfx820/r               | N/A       | $159.94   | 0.85       | Both listings use the identical Sony model number DVP-FX820 and describe the same 8" portable DVD player features; the only difference is color variant (blue vs red), which is a minor attribute variat... |
| 14  | 532       | sony white 8 ' portable dvd player dvpfx820w                    | 866       | sony dvp-fx820 / p portable dvd player dvpfx820/p               | N/A       | $148.72   | 0.85       | Both listings reference Sony DVP-FX820 portable DVD player with 8" screen; model numbers match despite different color variants (white vs pink).                                                            |
| 15  | 559       | panasonic viera 50 ' 1080p plasma hdtv in black th50pz850u      | 558       | panasonic viera th-50pz85u 50 ' plasma tv                       | N/A       | $1499.00  | 0.93       | Both listings describe a Panasonic Viera 50" 1080p plasma TV with the same model family (TH-50PZ850U). The names differ only by minor typographical variations, and the specifications (size, resolution... |
| 16  | 588       | belkin cush top for computer laptop f8n044slv                   | 230       | belkin cushtop f8n044-grn                                       | N/A       | N/A       | 0.92       | Both listings use the Belkin CushTop model f8n044; the only difference is the color suffix (slv vs grn), indicating the same product in different color variants.                                           |
| 17  | 589       | belkin cush top for computer laptop f8n044grn                   | 229       | belkin cushtop f8n044-org                                       | N/A       | N/A       | 0.85       | Both listings use the same Belkin model number f8n044, differing only in color suffix (green vs orange), indicating they are color variants of the same product.                                            |
| 18  | 654       | panasonic viera 50 ' plasma flat panel black hdtv th50pz800u    | 561       | panasonic viera th-50pz80u 50 ' plasma tv                       | N/A       | $1364.96  | 0.93       | Both listings describe a Panasonic Viera 50" plasma HDTV with the same model family (TH-50PZ800U vs TH-50PZ80U, likely a typo). Specifications and screen size match, and the price is reasonable for th... |
| 19  | 673       | lg stainless steel freestanding electric range lre30453ss       | 785       | lg 5.6 cu.ft . freestanding electric range                      | N/A       | N/A       | 0.85       | Both listings describe an LG freestanding electric range with 5.6 cu ft capacity; Product A provides the specific model LRE30453SS, which aligns with the generic description in Product B, indicating t... |
| 20  | 679       | tomtom one xl 330 car gps navigation system 1eg005200           | 930       | tomtom xl 330 s portable gps sysytem text to speech 4.3 ' to... | $246.95   | N/A       | 0.85       | Both listings reference a TomTom XL 330 with a 4.3" touchscreen and similar model numbers (1eg005200 vs garbled 1eg0.052.01), indicating they are the same product despite wording differences.             |
| 21  | 689       | lg black freestanding electric range lre30757bk                 | 786       | lg 5.6 cu.ft . freestanding electric range                      | N/A       | N/A       | 0.75       | Both listings describe an LG freestanding electric range with a 5.6 cu ft capacity; the detailed model LRE30757BK in Product A matches the generic description in Product B, suggesting they refer to th... |
| 22  | 690       | lg stainless steel freestanding electric range lre30757ss       | 785       | lg 5.6 cu.ft . freestanding electric range                      | N/A       | N/A       | 0.78       | Both listings describe an LG freestanding electric range with a 5.6 cu ft capacity; the detailed model LRE30757SS in Product A matches the generic description in Product B, suggesting they refer to th... |
| 23  | 691       | lg 5.6 cu . ft. white freestanding electric range lre30453wh    | 786       | lg 5.6 cu.ft . freestanding electric range                      | N/A       | N/A       | 0.85       | Both listings specify LG brand, 5.6 cu ft freestanding electric range; product A provides the exact model LRE30453WH which fits the generic description of product B, indicating they refer to the same ... |
| 24  | 692       | lg 5.6 cu . ft. black freestanding electric range lre30453bk    | 785       | lg 5.6 cu.ft . freestanding electric range                      | N/A       | N/A       | 0.7        | Both listings describe an LG freestanding electric range with a 5.6 cu ft capacity. Product A provides the specific model LRE30453BK and color, while Product B is a generic title lacking those details... |
| 25  | 698       | lg lfc25770st 25.0 cu . ft. stainless steel french door bott... | 772       | lg 24.7 cu . ft. stainless steel french door bottom mount re... | N/A       | N/A       | 0.85       | Both listings describe an LG stainless steel French‑door refrigerator with a bottom freezer and a capacity around 25 cu ft. The slight difference in reported capacity (25.0 vs 24.7 cu ft) is within ty... |
| 26  | 714       | lg 30 ' stainless steel freestanding gas range lrg30357ss       | 787       | lg 5.0 cu.ft . freestanding gas range                           | N/A       | N/A       | 0.63       | Both listings describe an LG freestanding gas range with a 5.0 cu ft capacity. Product A provides the full model (LRG30357SS) and detailed features, while Product B gives a generic name that matches t... |
| 27  | 714       | lg 30 ' stainless steel freestanding gas range lrg30357ss       | 789       | lg 5.0 cu.ft . freestanding gas range                           | N/A       | N/A       | 0.78       | Both listings describe an LG freestanding gas range with a 5.0 cu ft capacity; the generic name in B matches the detailed name and specs in A, and there is no conflicting information.                     |
| 28  | 716       | lg over-the-range white microwave oven lmv1680wh                | 780       | lg 1.6 cu.ft . over the range microwave                         | N/A       | N/A       | 0.85       | Both are LG over‑the‑range microwaves and the LMV1680WH model is a 1.6 cu ft unit, matching the capacity mentioned in Product B; despite missing model details in B, the description aligns.                |
| 29  | 727       | omnimount stellar series audio tower g303dark                   | 257       | omnimount 3-shelf large-component tower stellarg303g            | $299.00   | N/A       | 0.92       | Both listings use Omnimount, mention a stellar series tower with model g303 (dark vs g303g) and a 3-shelf large-component design, indicating they refer to the same product despite naming variations.      |
| 30  | 763       | blueant black bluetooth headset z9i                             | 958       | blueant bluetooth hs dual mic nic 091004                        | $99.00    | $99.99    | 0.7        | Both are BlueAnt Bluetooth headsets with dual microphones, similar price, and overlapping branding; although model identifiers differ, the lack of distinct specs suggests they refer to the same produc... |
| 31  | 769       | garmin nuvi 205 gps navigation system 0100071740                | 947       | garmin auto nav 010-00715-20                                    | N/A       | $410.95   | 0.7        | Both listings refer to a Garmin navigation device with very similar part numbers (0100071740 vs 010-00715-20) and the name Nuvi 205 appears in one; the other uses a generic auto nav label but likely t... |
| 32  | 794       | speck clear 13 ' macbook see thru hard shell case mb13clrsee... | 677       | speck products seethru case for apple macbook air mba-clr-se... | N/A       | N/A       | 0.85       | Both listings describe a Speck clear, see‑through hard plastic case for a 13‑inch MacBook Air, with similar model identifiers (mb13clrseev2 vs mba-clr-see) indicating the same product line; descriptio... |
| 33  | 840       | speck seethru pink hard shell case for 13 ' macbook mb13pnks... | 676       | speck products seethru case for apple macbook air mba-pnk-se... | N/A       | N/A       | 0.86       | Both listings describe a Speck Seethru pink case for a 13" MacBook Air, with similar model identifiers (mb13pnkseev2 vs mba-pnk-see) and matching color and style, indicating they refer to the same pro... |

#### False Negatives (8)

| #   | Source ID | Source Name                                                 | Target ID | Target Name                                                     | Src Price | Tgt Price | Confidence | Reasoning                                                                                                                                                                                                   |
| --- | --------- | ----------------------------------------------------------- | --------- | --------------------------------------------------------------- | --------- | --------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | 245       | lg wm3431w all-in-one white washer and dryer combo wm3431wh | 215       | lg electronics 24 ' washer/dryer combo : white                  | N/A       | N/A       | 0.4        | Product A provides a specific model number and detailed specs, while Product B is a generic description lacking model information, making it impossible to confirm they are the same item.                  |
| 2   | 36        | cuisinart automatic brew and serve coffeemaker dtc975bk     | 228       | cuisinart brew & serve stainless therm black 10-cup coffee m... | $99.95    | $99.95    | 0.75       | Both are Cuisinart brew & serve coffee makers with similar branding and price, but Product A specifies model DTC975BK with a 12‑cup insulated carafe, while Product B mentions a 10‑cup version and lack... |
| 3   | 41        | maytag bisque over-the-range microwave oven mmv4205bt       | 811       | maytag mmv4205baq over-the-range microwave                      | N/A       | N/A       | 0.85       | The model numbers differ (mmv4205bt vs mmv4205baq) indicating different variants, and only one mentions a bisque finish, suggesting they are not the identical product.                                     |
| 4   | 410       | apple 500gb time capsule wireless hard drive mb276lla       | 645       | apple time capsule network hard drive mb277ll/a                 | $299.00   | $439.00   | 0.96       | Different model numbers (MB276LL/A vs MB277LL/A), different storage capacities (500 GB vs 1 TB), and price discrepancy indicate they are distinct Apple Time Capsule products.                              |
| 5   | 45        | whirlpool 24 ' built-in dishwasher du1100ss                 | 1085      | whirlpool du1100xtps 24 ' undercounter dishwasher ( stainles... | N/A       | $537.29   | 0.85       | Although both are 24" Whirlpool undercounter dishwashers, the model numbers differ (du1100ss vs du1100xtps) indicating distinct variants, and there is no matching description to confirm they are the s... |
| 6   | 55        | delonghi twenty four seven coffee maker dc50w               | 225       | delonghi 4-cup drip coffee makers                               | $22.00    | N/A       | 0.6        | Product A specifies the exact model (DC50W) and details, while Product B is a generic listing for any Delonghi 4‑cup drip coffee maker, lacking model info; the overlap is insufficient to confirm they ... |
| 7   | 626       | polk audio csi a4 black center channel loudspeaker csia4bk  | 471       | polkaudio csi a4 black high performance center channel louds... | $279.95   | $189.47   | 0.85       | Both listings mention a Polk Audio CSI A4 black center channel, but they use different model numbers (CSIA4BK vs AM4415-A) and the price gap is large, indicating they are different products within the... |
| 8   | 632       | polk audio black 10 ' powered subwoofer psw110bk            | 565       | polkaudio psw series psw110 powered subwoofer                   | $299.95   | $249.95   | 0.85       | Although the names reference the same Polk Audio PSW110 model, Product B's description only mentions a "woofer cable" and its price is lower, indicating it is likely an accessory rather than the subwo... |

### Multi-Agent

| Metric    | Count    |
| --------- | -------- |
| TP        | 192      |
| FP        | 26       |
| FN        | 14       |
| TN        | 1684     |
| **Total** | **1916** |
| Precision | 0.8807   |
| Recall    | 0.9320   |
| F1        | 0.9057   |

#### False Positives (26)

| #   | Source ID | Source Name                                                     | Target ID | Target Name                                                     | Src Price | Tgt Price | Confidence | Reasoning                                                                                                                                                                                                   |
| --- | --------- | --------------------------------------------------------------- | --------- | --------------------------------------------------------------- | --------- | --------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | 1007      | lg white xl load capacity electric dryer dle0442wh              | 424       | lg xl capacity electric dryer                                   | N/A       | N/A       | 0.86       | Both listings refer to an LG electric dryer with XL capacity. Product B’s name is a concise version of Product A’s name, and the core identifiers (LG, XL, capacity, electric dryer) match exactly. The ... |
| 2   | 1030      | transcend 8gb micro secure digital memory card ts8gusdhc6       | 333       | transcend 8gb secure digital high capacity ( sdhc ) card cla... | $39.00    | $11.45    | 0.86       | Both listings describe an 8 GB Transcend micro SDHC card, class 6, with essentially the same model identifier (minor typo). The only notable differences are the inclusion of an adapter in Product A an... |
| 3   | 110       | canon photo ink cartridge cl52                                  | 84        | canon ink cartridge for pixma ip1600 , ip6210d and ip6220d p... | $25.00    | $19.11    | 0.85       | Both listings refer to Canon's CL‑52 color ink cartridge. The part number 0617B002 listed for Product B is the official part number for the CL‑52 cartridge, matching the model described in Product A. ... |
| 4   | 175       | tech craft avalon series tv stand swp48                         | 279       | abs48 wood tv stand ( avalon series , 48-in . max )             | $299.00   | N/A       | 0.85       | Both listings contain the distinctive identifiers 'Avalon Series', 'TV stand', and size '48' (or 48‑in). The model code SWP48 in Product A aligns with the 'abs48' reference in Product B, indicating th... |
| 5   | 292       | canon black ink cartridge pg50                                  | 88        | canon black ink cartridge 0615b002                              | $29.00    | N/A       | 0.9        | Both products share the core name 'canon black ink cartridge' and the model identifiers PG50 and 0615B002 correspond to the same Canon PG‑50 black ink cartridge (0615B002 is the OEM part number for PG... |
| 6   | 37        | sharp over the counter microwave oven r1214ss                   | 394       | sharp 1100 watt over the counter microwave                      | $429.00   | N/A       | 0.66       | Both listings describe a Sharp over‑the‑counter microwave with 1100 W power. Product A provides a specific model (r1214ss) and detailed specs, while Product B gives only a generic name that matches th... |
| 7   | 444       | sony silver cyber-shot 7.2 megapixel digital camera dscw120     | 573       | sony cyber-shot dsc-w120 digital camera pink dscw120/p          | N/A       | $109.99   | 0.82       | Both listings reference the identical Sony Cyber-shot DSC‑W120 model number and share the same key specifications (2.5" TFT LCD, 2x digital zoom, etc.). The only discrepancy is the color (silver vs pi... |
| 8   | 486       | canon silver 8.0 megapixel powershot digital camera sd1100is    | 543       | canon powershot sd1100 is digital elph camera melody pink 25... | N/A       | N/A       | 0.78       | Both listings identify the Canon PowerShot SD1100 IS camera with identical core specifications (8 MP, 3× optical zoom). The only discrepancy is the color/finish (silver vs. pink melody), which represe... |
| 9   | 486       | canon silver 8.0 megapixel powershot digital camera sd1100is    | 544       | canon powershot sd1100 is digital elph camera rhythm & blue ... | N/A       | N/A       | 0.81       | Both listings identify the Canon PowerShot SD1100 IS model, specifying the same 8 MP sensor and 3× optical zoom. The only differences are color/finish descriptors (silver vs rhythm & blue) and extra S... |
| 10  | 501       | panasonic viera 50 ' plasma flat panel 1080p hdtv in black t... | 556       | panasonic viera th-50px80u 50 ' plasma tv                       | N/A       | N/A       | 0.71       | Both listings identify a Panasonic Viera 50" plasma TV. The model identifiers are extremely similar (th50pz80u vs th-50px80u) and likely represent typographical or regional variations of the same mode... |
| 11  | 504       | panasonic viera 46 ' plasma flat panel 1080p hdtv in black t... | 843       | panasonic viera th-46pz850u 46 ' plasma tv th46pz850u           | N/A       | $1695.96  | 0.93       | Both listings reference a Panasonic Viera 46" plasma HDTV with essentially the same model identifier (th46pz85u vs th-46pz850u), and the specifications (46", 1080p, plasma) align. The minor discrepanc... |
| 12  | 520       | nikon coolpix s550 10 megapixel black digital camera coolpix... | 538       | nikon coolpix s550 digital camera cool blue 26109               | N/A       | $169.95   | 0.86       | Both listings reference the Nikon Coolpix S550 model, sharing the core identifier and key specs (10 MP, 2.5" LCD). The only differences are color (black vs cool blue) and how zoom is described (5× opt... |
| 13  | 531       | sony dvp-fx820 red 8 ' portable dvd player dvpfx820r            | 879       | sony dvpfx820 portable dvd player dvpfx820/w                    | N/A       | $149.00   | 0.88       | Both listings reference the identical Sony DVP-FX820 model number, and all core specifications align. The only discrepancy is the color (red vs. white), which is a typical variant attribute rather tha... |
| 14  | 559       | panasonic viera 50 ' 1080p plasma hdtv in black th50pz850u      | 558       | panasonic viera th-50pz85u 50 ' plasma tv                       | N/A       | $1499.00  | 0.91       | Both listings describe a 50" Panasonic Viera plasma HDTV with 1080p resolution, identical feature sets, and nearly identical model identifiers (TH‑50PZ850U vs TH‑50PZ85U, the latter likely a typo). Th... |
| 15  | 654       | panasonic viera 50 ' plasma flat panel black hdtv th50pz800u    | 561       | panasonic viera th-50pz80u 50 ' plasma tv                       | N/A       | $1364.96  | 0.91       | Both listings refer to a Panasonic Viera 50" plasma HDTV with the same resolution and model family. The model identifiers differ only by a minor typographical variation (th50pz800u vs th-50pz80u), whi... |
| 16  | 673       | lg stainless steel freestanding electric range lre30453ss       | 785       | lg 5.6 cu.ft . freestanding electric range                      | N/A       | N/A       | 0.78       | Both listings refer to an LG freestanding electric range with a 5.6 cu ft capacity. Product A provides the exact model number (LRE30453SS) and detailed specs, while Product B offers a shortened, gener... |
| 17  | 689       | lg black freestanding electric range lre30757bk                 | 786       | lg 5.6 cu.ft . freestanding electric range                      | N/A       | N/A       | 0.85       | Both products are LG freestanding electric ranges with the same 5.6 cu ft capacity. Product A provides the specific model number LRE30757BK, which aligns with the generic description of Product B. No ... |
| 18  | 690       | lg stainless steel freestanding electric range lre30757ss       | 785       | lg 5.6 cu.ft . freestanding electric range                      | N/A       | N/A       | 0.75       | Both listings share the core attributes: brand (LG), product type (freestanding electric range), and capacity (5.6 cu ft). The detailed listing (Product A) provides the specific model LRE30757SS and s... |
| 19  | 691       | lg 5.6 cu . ft. white freestanding electric range lre30453wh    | 786       | lg 5.6 cu.ft . freestanding electric range                      | N/A       | N/A       | 0.86       | Both entries describe an LG freestanding electric range with a 5.6 cu ft capacity. Product A provides additional details (white finish, model LRE30453WH) that are consistent with the generic descripti... |
| 20  | 692       | lg 5.6 cu . ft. black freestanding electric range lre30453bk    | 785       | lg 5.6 cu.ft . freestanding electric range                      | N/A       | N/A       | 0.91       | Both entries describe an LG freestanding electric range with a 5.6 cu ft capacity. Product A adds specific details (model LRE30453BK, black finish) that are consistent with the generic description in ... |
| 21  | 698       | lg lfc25770st 25.0 cu . ft. stainless steel french door bott... | 772       | lg 24.7 cu . ft. stainless steel french door bottom mount re... | N/A       | N/A       | 0.82       | Both listings describe an LG stainless‑steel French‑door refrigerator with a bottom freezer/freezer compartment and a capacity of roughly 25 cu ft (24.7 vs 25.0). The model identifier appears only in ... |
| 22  | 714       | lg 30 ' stainless steel freestanding gas range lrg30357ss       | 787       | lg 5.0 cu.ft . freestanding gas range                           | N/A       | N/A       | 0.66       | Both entries share the core attributes—brand LG, freestanding gas range, and 5.0 cu ft capacity. Product A provides detailed specs (30" stainless steel, model LRG30357SS) that are consistent with the ... |
| 23  | 714       | lg 30 ' stainless steel freestanding gas range lrg30357ss       | 789       | lg 5.0 cu.ft . freestanding gas range                           | N/A       | N/A       | 0.66       | Both listings are for an LG freestanding gas range with a 5.0 cu ft capacity. Product A provides detailed attributes (30" stainless steel, model LRG30357SS) that are consistent with the generic descri... |
| 24  | 716       | lg over-the-range white microwave oven lmv1680wh                | 780       | lg 1.6 cu.ft . over the range microwave                         | N/A       | N/A       | 0.75       | Both listings refer to an LG over‑the‑range microwave with a 1.6 cu ft capacity. The detailed listing (Product A) specifies the exact model LMV1680WH, which is a white, 1.6 cu ft over‑the‑range unit. ... |
| 25  | 727       | omnimount stellar series audio tower g303dark                   | 257       | omnimount 3-shelf large-component tower stellarg303g            | $299.00   | N/A       | 0.75       | Both listings share the brand Omnimount and the model identifier G303 (appearing as 'g303dark' and 'stellarg303g'). The product names and the limited description both indicate a three‑shelf tower for ... |
| 26  | 965       | sanus 30 ' 58 ' visionmount flat panel tv black tilting wall... | 160       | sanus visionmount tilting flat panel tv wall mount mt25-b1      | $199.00   | N/A       | 0.81       | Both listings share the same brand (Sanus), product line (VisionMount), and key attributes (tilting flat‑panel TV wall mount). The only discrepancy is the model code (lt25b1 vs mt25‑b1), which is like... |

#### False Negatives (14)

| #   | Source ID | Source Name                                                 | Target ID | Target Name                                                     | Src Price | Tgt Price | Confidence | Reasoning                                                                                                                                                                                                   |
| --- | --------- | ----------------------------------------------------------- | --------- | --------------------------------------------------------------- | --------- | --------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | 121       | tivo wireless adapter ag0100                                | 371       | tvio wireless usb network adpator                               | N/A       | N/A       | 0.75       | Both agents conclude no match; the first product includes a specific model number (ag0100) and is explicitly for TiVo Series 2, while the second lacks this identifier and uses generic terms like 'usb ... |
| 2   | 122       | uniden 5.8 ghz accessory handset and charger tcx905         | 137       | uniden tcx905 tru-digital expansion handset with caller id      | N/A       | $29.99    | 0.73       | Both listings refer to a Uniden TCX905 expansion handset, but Product A explicitly includes a charger and a longer feature list, indicating a bundled accessory set, whereas Product B describes only th... |
| 3   | 245       | lg wm3431w all-in-one white washer and dryer combo wm3431wh | 215       | lg electronics 24 ' washer/dryer combo : white                  | N/A       | N/A       | 0.71       | Product A provides a specific model number (WM3431W/WM3431WH) and detailed specifications, while Product B only gives a generic name ('lg electronics 24' washer/dryer combo : white') with no model ide... |
| 4   | 281       | tech craft dark cherry veneto series tv stand swp60         | 280       | techcraft swp60 classic wood tone credenza tv stand             | $399.00   | N/A       | 0.78       | Both listings share the brand and model identifier SWP60, but the product names and descriptions indicate different finishes (dark cherry veneer vs walnut wood) and the second listing lacks detailed s... |
| 5   | 36        | cuisinart automatic brew and serve coffeemaker dtc975bk     | 228       | cuisinart brew & serve stainless therm black 10-cup coffee m... | $99.95    | $99.95    | 0.85       | The two listings differ in model identifier and key specifications: Product A is a 12‑cup DTC975BK with a double‑wall insulated stainless steel carafe and extensive programmable features, while Produc... |
| 6   | 41        | maytag bisque over-the-range microwave oven mmv4205bt       | 811       | maytag mmv4205baq over-the-range microwave                      | N/A       | N/A       | 0.81       | Both agents agree the items are closely related, but the model numbers differ (mmv4205bt vs mmv4205baq) and the first explicitly mentions a bisque finish while the second does not, indicating distinct... |
| 7   | 410       | apple 500gb time capsule wireless hard drive mb276lla       | 645       | apple time capsule network hard drive mb277ll/a                 | $299.00   | $439.00   | 0.92       | Both agents correctly identified key distinguishing attributes: different Apple model numbers (MB276LL/A vs MB277LL/A) and different storage capacities (500 GB vs 1 TB). The price discrepancy further ... |
| 8   | 45        | whirlpool 24 ' built-in dishwasher du1100ss                 | 1085      | whirlpool du1100xtps 24 ' undercounter dishwasher ( stainles... | N/A       | $537.29   | 0.78       | Both listings share brand, size, and dishwasher type, but the model numbers differ (du1100ss vs du1100xtps). In Whirlpool's naming convention, different suffixes usually denote distinct SKUs with diff... |
| 9   | 55        | delonghi twenty four seven coffee maker dc50w               | 225       | delonghi 4-cup drip coffee makers                               | $22.00    | N/A       | 0.78       | Product A specifies a concrete model (twenty‑four‑seven DC50W) with detailed attributes, while Product B is a generic, plural listing for any Delonghi 4‑cup drip coffee maker and lacks model details o... |
| 10  | 626       | polk audio csi a4 black center channel loudspeaker csia4bk  | 471       | polkaudio csi a4 black high performance center channel louds... | $279.95   | $189.47   | 0.75       | Both listings share the brand, series (CSI A4), color, and speaker type, which suggests they could be the same product. However, the model identifiers differ (CSIA4BK vs AM4415-A), and the price gap (... |
| 11  | 632       | polk audio black 10 ' powered subwoofer psw110bk            | 565       | polkaudio psw series psw110 powered subwoofer                   | $299.95   | $249.95   | 0.92       | Product A is a complete 10" powered subwoofer with detailed specs, while Product B's description is merely 'woofer cable', indicating an accessory. The mismatch in product type, description, and purpo... |
| 12  | 814       | sony bud style headphones in silver mdred12lpslv            | 702       | bass lover earbuds slver mdred12lp/slv                          | $14.00    | $10.99    | 0.88       | Both agents correctly note that the brand and product type differ (Sony headphones vs Bass Lover earbuds) despite a shared model code fragment. The detailed description for Sony and lack of descriptio... |
| 13  | 876       | nikon sb-900 af speedlight in black sb900                   | 931       | sb-900 af speedlight w / - stnd diffsn dome flt set 4807        | $499.00   | $438.88   | 0.86       | Both listings refer to the Nikon SB‑900 AF speedlight, but Product B explicitly includes a 'standard diffusion dome filter set 4807' accessory, which is not mentioned in Product A. The presence of thi... |
| 14  | 91        | canon cyan ink tank cyan cli8c                              | 86        | canon cli-8c ink cartridge 0621b002                             | $16.00    | $13.99    | 0.85       | Both listings reference the Canon CLI-8C cyan cartridge, but Product A's description explicitly calls it a 'compatible' ink tank, implying a third‑party replacement, while Product B includes the OEM p... |

## 3. Error Category Classification

### Single LLM

#### False Positive Categories (33 total)

| Category            | Count | %     |
| ------------------- | ----- | ----- |
| color variant       | 13    | 39.4% |
| accessory confusion | 8     | 24.2% |
| other               | 8     | 24.2% |
| form factor         | 2     | 6.1%  |
| generic vs specific | 1     | 3.0%  |
| model variant       | 1     | 3.0%  |

#### False Negative Categories (8 total)

| Category           | Count | %     |
| ------------------ | ----- | ----- |
| sparse description | 6     | 75.0% |
| price mismatch     | 1     | 12.5% |
| name mismatch      | 1     | 12.5% |

### Multi-Agent

#### False Positive Categories (26 total)

| Category            | Count | %     |
| ------------------- | ----- | ----- |
| color variant       | 8     | 30.8% |
| other               | 7     | 26.9% |
| accessory confusion | 6     | 23.1% |
| model variant       | 3     | 11.5% |
| generic vs specific | 1     | 3.8%  |
| form factor         | 1     | 3.8%  |

#### False Negative Categories (14 total)

| Category           | Count | %     |
| ------------------ | ----- | ----- |
| sparse description | 11    | 78.6% |
| price mismatch     | 2     | 14.3% |
| name mismatch      | 1     | 7.1%  |

## 4. Multi-Agent Architecture Analysis

**Total pairs with agent data:** 1916

### Agent Agreement

| Metric           | Count | %     |
| ---------------- | ----- | ----- |
| Agents agreed    | 1867  | 97.4% |
| Agents disagreed | 49    | 2.6%  |

### When Agents Disagreed

- **Syntactic agent correct:** 36 (73.5%)
- **Semantic agent correct:** 13 (26.5%)

### Orchestrator Override of Agent Consensus

- **Override count:** 3
- **Override correct:** 2 (66.7%)

### Per-Agent Performance

| Metric    | Syntactic Agent | Semantic Agent |
| --------- | --------------- | -------------- |
| TP        | 193             | 175            |
| FP        | 28              | 33             |
| FN        | 13              | 31             |
| TN        | 1682            | 1677           |
| Precision | 0.8733          | 0.8413         |
| Recall    | 0.9369          | 0.8495         |
| F1        | 0.9040          | 0.8454         |

### Unique Contributions

- **Syntactic correctly identified match when semantic missed:** 21
- **Semantic correctly identified match when syntactic missed:** 3
- **Syntactic correctly rejected when semantic false-alarmed:** 15
- **Semantic correctly rejected when syntactic false-alarmed:** 10

## 5. Cross-Pipeline Disagreement Analysis

**Total disagreements:** 25

- **Single LLM correct (multi-agent wrong):** 12
- **Multi-agent correct (single LLM wrong):** 13

### Detailed Disagreements

| #   | Source Name                                      | Target Name                                      | True     | Single Verdict  | Multi Verdict   | Correct |
| --- | ------------------------------------------------ | ------------------------------------------------ | -------- | --------------- | --------------- | ------- |
| 1   | lg white xl load capacity electric dryer dle0... | lg xl capacity electric dryer                    | NO MATCH | NO MATCH (0.6)  | MATCH (0.86)    | Single  |
| 2   | canon eos rebel xsi silver digital slr camera... | canon eos rebel xsi digital slr camera with e... | NO MATCH | MATCH (0.85)    | NO MATCH (0.78) | Multi   |
| 3   | tivo wireless adapter ag0100                     | tvio wireless usb network adpator                | MATCH    | MATCH (0.7)     | NO MATCH (0.75) | Single  |
| 4   | uniden 5.8 ghz accessory handset and charger ... | uniden tcx905 tru-digital expansion handset w... | MATCH    | MATCH (0.92)    | NO MATCH (0.73) | Single  |
| 5   | sony progressive scan silver dvd player dvpns... | sony dvpns57p dvd player dvpns57pb               | NO MATCH | MATCH (0.85)    | NO MATCH (0.81) | Multi   |
| 6   | tech craft dark cherry veneto series tv stand... | techcraft swp60 classic wood tone credenza tv... | MATCH    | MATCH (0.7)     | NO MATCH (0.78) | Single  |
| 7   | canon black ink cartridge pg50                   | canon black ink cartridge 0615b002               | NO MATCH | NO MATCH (0.85) | MATCH (0.9)     | Single  |
| 8   | sharp over the counter microwave oven r1214ss    | sharp 1100 watt over the counter microwave       | NO MATCH | NO MATCH (0.6)  | MATCH (0.66)    | Single  |
| 9   | lg dlex8377nm navy blue xl capacity electric ... | lg 27 ' front-load electric dryer with 7.3 cu... | NO MATCH | MATCH (0.65)    | NO MATCH (0.85) | Multi   |
| 10  | canon blue 8.0 megapixel powershot digital ca... | canon powershot sd1100 is digital elph camera... | NO MATCH | MATCH (0.93)    | NO MATCH (0.75) | Multi   |
| 11  | panasonic viera 50 ' plasma flat panel 1080p ... | panasonic viera th-50px80u 50 ' plasma tv        | NO MATCH | NO MATCH (0.85) | MATCH (0.71)    | Single  |
| 12  | sony dvp-fx820 blue 8 ' portable dvd player d... | sony dvp-fx820 / r portable dvd player dvpfx8... | NO MATCH | MATCH (0.85)    | NO MATCH (0.78) | Multi   |
| 13  | sony dvp-fx820 red 8 ' portable dvd player dv... | sony dvpfx820 portable dvd player dvpfx820/w     | NO MATCH | NO MATCH (0.7)  | MATCH (0.88)    | Single  |
| 14  | sony white 8 ' portable dvd player dvpfx820w     | sony dvp-fx820 / p portable dvd player dvpfx8... | NO MATCH | MATCH (0.85)    | NO MATCH (0.75) | Multi   |
| 15  | belkin cush top for computer laptop f8n044slv    | belkin cushtop f8n044-grn                        | NO MATCH | MATCH (0.92)    | NO MATCH (0.78) | Multi   |
| 16  | belkin cush top for computer laptop f8n044grn    | belkin cushtop f8n044-org                        | NO MATCH | MATCH (0.85)    | NO MATCH (0.85) | Multi   |
| 17  | tomtom one xl 330 car gps navigation system 1... | tomtom xl 330 s portable gps sysytem text to ... | NO MATCH | MATCH (0.85)    | NO MATCH (0.78) | Multi   |
| 18  | blueant black bluetooth headset z9i              | blueant bluetooth hs dual mic nic 091004         | NO MATCH | MATCH (0.7)     | NO MATCH (0.81) | Multi   |
| 19  | garmin nuvi 205 gps navigation system 0100071... | garmin auto nav 010-00715-20                     | NO MATCH | MATCH (0.7)     | NO MATCH (0.92) | Multi   |
| 20  | speck clear 13 ' macbook see thru hard shell ... | speck products seethru case for apple macbook... | NO MATCH | MATCH (0.85)    | NO MATCH (0.82) | Multi   |
| 21  | sony bud style headphones in silver mdred12lp... | bass lover earbuds slver mdred12lp/slv           | MATCH    | MATCH (0.92)    | NO MATCH (0.88) | Single  |
| 22  | speck seethru pink hard shell case for 13 ' m... | speck products seethru case for apple macbook... | NO MATCH | MATCH (0.86)    | NO MATCH (0.92) | Multi   |
| 23  | nikon sb-900 af speedlight in black sb900        | sb-900 af speedlight w / - stnd diffsn dome f... | MATCH    | MATCH (0.92)    | NO MATCH (0.86) | Single  |
| 24  | canon cyan ink tank cyan cli8c                   | canon cli-8c ink cartridge 0621b002              | MATCH    | MATCH (0.93)    | NO MATCH (0.85) | Single  |
| 25  | sanus 30 ' 58 ' visionmount flat panel tv bla... | sanus visionmount tilting flat panel tv wall ... | NO MATCH | NO MATCH (0.92) | MATCH (0.81)    | Single  |

### Disagreement Details

**1. lg white xl load capacity electric dryer dle0442wh vs lg xl capacity electric dryer**

- True label: NO MATCH | Correct: **Single LLM**
- Single LLM: NO MATCH (conf 0.6): Product A is a specific LG model (DLE0442WH) with detailed specs, while Product B is a generic description lacking model number or details, making it impossible to confirm they refer to the same item.
- Multi-Agent: MATCH (conf 0.86): Both listings refer to an LG electric dryer with XL capacity. Product B’s name is a concise version of Product A’s name, and the core identifiers (LG, XL, capacity, electric dryer) match exactly. The ...

**2. canon eos rebel xsi silver digital slr camera xsireb1855s vs canon eos rebel xsi digital slr camera with ef-s 18-55mm f/3 .5 -5.6 is lens black 2756b003**

- True label: NO MATCH | Correct: **Multi-Agent**
- Single LLM: MATCH (conf 0.85): Both listings describe the Canon EOS Rebel XSi digital SLR with the same model number (2756B003) and identical core specs (12 MP, 18‑55mm lens). The only difference is color and price, which are typic...
- Multi-Agent: NO MATCH (conf 0.78): Both listings refer to the Canon EOS Rebel XSi kit (model 2756B003) with an 18‑55mm lens, but they differ on a key attribute—color (silver vs. black)—and have a noticeable price gap. In product matchi...

**3. tivo wireless adapter ag0100 vs tvio wireless usb network adpator**

- True label: MATCH | Correct: **Single LLM**
- Single LLM: MATCH (conf 0.7): Names are very similar despite typos and both refer to a Tivo wireless USB adapter; the specific model AG0100 in Product A suggests a Tivo series 2 adapter, which aligns with the generic description o...
- Multi-Agent: NO MATCH (conf 0.75): Both agents conclude no match; the first product includes a specific model number (ag0100) and is explicitly for TiVo Series 2, while the second lacks this identifier and uses generic terms like 'usb ...

**4. uniden 5.8 ghz accessory handset and charger tcx905 vs uniden tcx905 tru-digital expansion handset with caller id**

- True label: MATCH | Correct: **Single LLM**
- Single LLM: MATCH (conf 0.92): Both listings reference the Uniden TCX905 handset accessory; product names and key features (handset, caller ID) align, and price information does not contradict a match.
- Multi-Agent: NO MATCH (conf 0.73): Both listings refer to a Uniden TCX905 expansion handset, but Product A explicitly includes a charger and a longer feature list, indicating a bundled accessory set, whereas Product B describes only th...

**5. sony progressive scan silver dvd player dvpns57ps vs sony dvpns57p dvd player dvpns57pb**

- True label: NO MATCH | Correct: **Multi-Agent**
- Single LLM: MATCH (conf 0.85): Both listings reference Sony DVP-NS57P progressive scan DVD player with identical model base; suffixes differ only by color (silver vs black), indicating variants of the same product.
- Multi-Agent: NO MATCH (conf 0.81): Both listings refer to the Sony DVP‑NS57P DVD player model, but the suffixes (ps vs pb) and description details indicate different finishes (silver vs black). Color/finish variants are treated as dist...

**6. tech craft dark cherry veneto series tv stand swp60 vs techcraft swp60 classic wood tone credenza tv stand**

- True label: MATCH | Correct: **Single LLM**
- Single LLM: MATCH (conf 0.7): Both listings share the unique model identifier SWP60 and similar product type (techcraft TV stand/credenza). The only differences are finish descriptors, suggesting variant colors of the same base pr...
- Multi-Agent: NO MATCH (conf 0.78): Both listings share the brand and model identifier SWP60, but the product names and descriptions indicate different finishes (dark cherry veneer vs walnut wood) and the second listing lacks detailed s...

**7. canon black ink cartridge pg50 vs canon black ink cartridge 0615b002**

- True label: NO MATCH | Correct: **Single LLM**
- Single LLM: NO MATCH (conf 0.85): Product A is a PG50 inkjet cartridge with specific pigment ink description, while Product B lists a different part number (0615B002) likely for a different cartridge type; the lack of matching model n...
- Multi-Agent: MATCH (conf 0.9): Both products share the core name 'canon black ink cartridge' and the model identifiers PG50 and 0615B002 correspond to the same Canon PG‑50 black ink cartridge (0615B002 is the OEM part number for PG...

**8. sharp over the counter microwave oven r1214ss vs sharp 1100 watt over the counter microwave**

- True label: NO MATCH | Correct: **Single LLM**
- Single LLM: NO MATCH (conf 0.6): Product B provides only a generic name and wattage, lacking the specific model number, capacity, and feature details that identify Product A. The shared brand and 1100 W rating are insufficient to con...
- Multi-Agent: MATCH (conf 0.66): Both listings describe a Sharp over‑the‑counter microwave with 1100 W power. Product A provides a specific model (r1214ss) and detailed specs, while Product B gives only a generic name that matches th...

**9. lg dlex8377nm navy blue xl capacity electric steamdryer dlex8377blue vs lg 27 ' front-load electric dryer with 7.3 cu . ft. capacity**

- True label: NO MATCH | Correct: **Multi-Agent**
- Single LLM: MATCH (conf 0.65): Both listings describe an LG front‑load electric dryer with 7.3 cu ft capacity; product A provides the specific model DLEX8377NM, which matches the generic description of product B, suggesting they re...
- Multi-Agent: NO MATCH (conf 0.85): Product A specifies the LG model DLEX8377NM, navy‑blue color, and steam‑dryer features, while Product B is a generic LG front‑load dryer listing with no model, color, or steam details. The lack of mat...

**10. canon blue 8.0 megapixel powershot digital camera sd1100isb vs canon powershot sd1100 is digital elph camera swing silver 2508b001**

- True label: NO MATCH | Correct: **Multi-Agent**
- Single LLM: MATCH (conf 0.93): Both listings describe the Canon PowerShot SD1100 IS 8 MP camera with 3× optical zoom; the only difference is the color (blue vs silver), which is a variant of the same model.
- Multi-Agent: NO MATCH (conf 0.75): Both listings describe the Canon Powershot SD1100 model with identical core specs (8 MP, 3× optical zoom). However, the model suffixes and descriptions indicate different color variants: 'sd1100isb' (...

**11. panasonic viera 50 ' plasma flat panel 1080p hdtv in black th50pz80u vs panasonic viera th-50px80u 50 ' plasma tv**

- True label: NO MATCH | Correct: **Single LLM**
- Single LLM: NO MATCH (conf 0.85): Model numbers differ (th50pz80u vs th-50px80u) and specifications conflict: A lists 1080p 1920x1080, B lists 1366x768, indicating different products.
- Multi-Agent: MATCH (conf 0.71): Both listings identify a Panasonic Viera 50" plasma TV. The model identifiers are extremely similar (th50pz80u vs th-50px80u) and likely represent typographical or regional variations of the same mode...

**12. sony dvp-fx820 blue 8 ' portable dvd player dvpfx820li vs sony dvp-fx820 / r portable dvd player dvpfx820/r**

- True label: NO MATCH | Correct: **Multi-Agent**
- Single LLM: MATCH (conf 0.85): Both listings use the identical Sony model number DVP-FX820 and describe the same 8" portable DVD player features; the only difference is color variant (blue vs red), which is a minor attribute variat...
- Multi-Agent: NO MATCH (conf 0.78): Both listings refer to the Sony DVP-FX820 model, but the suffixes ("LI" vs "/R") and explicit color mentions (blue vs red) indicate distinct SKU variants. The descriptions highlight different feature ...

**13. sony dvp-fx820 red 8 ' portable dvd player dvpfx820r vs sony dvpfx820 portable dvd player dvpfx820/w**

- True label: NO MATCH | Correct: **Single LLM**
- Single LLM: NO MATCH (conf 0.7): Both listings refer to the Sony DVP-FX820 series but specify different variants (red model dvpfx820r vs white model dvpfx820/w) and have distinct descriptions, indicating they are separate SKUs.
- Multi-Agent: MATCH (conf 0.88): Both listings reference the identical Sony DVP-FX820 model number, and all core specifications align. The only discrepancy is the color (red vs. white), which is a typical variant attribute rather tha...

**14. sony white 8 ' portable dvd player dvpfx820w vs sony dvp-fx820 / p portable dvd player dvpfx820/p**

- True label: NO MATCH | Correct: **Multi-Agent**
- Single LLM: MATCH (conf 0.85): Both listings reference Sony DVP-FX820 portable DVD player with 8" screen; model numbers match despite different color variants (white vs pink).
- Multi-Agent: NO MATCH (conf 0.75): Both listings refer to the Sony DVP-FX820 portable DVD player model, but they specify different colors (white vs pink) and distinct suffixes (w vs /p) that denote variant identifiers. Color is a mater...

**15. belkin cush top for computer laptop f8n044slv vs belkin cushtop f8n044-grn**

- True label: NO MATCH | Correct: **Multi-Agent**
- Single LLM: MATCH (conf 0.92): Both listings use the Belkin CushTop model f8n044; the only difference is the color suffix (slv vs grn), indicating the same product in different color variants.
- Multi-Agent: NO MATCH (conf 0.78): Both listings refer to the same Belkin cush top model (f8n044) but specify different color suffixes (slv for silver vs grn for green). In product entity resolution, color variants are treated as disti...

**16. belkin cush top for computer laptop f8n044grn vs belkin cushtop f8n044-org**

- True label: NO MATCH | Correct: **Multi-Agent**
- Single LLM: MATCH (conf 0.85): Both listings use the same Belkin model number f8n044, differing only in color suffix (green vs orange), indicating they are color variants of the same product.
- Multi-Agent: NO MATCH (conf 0.85): Both listings refer to the same Belkin cush top model (f8n044) but specify different colors—green in Product A and orange in Product B. Color variants are treated as distinct SKUs, and the description...

**17. tomtom one xl 330 car gps navigation system 1eg005200 vs tomtom xl 330 s portable gps sysytem text to speech 4.3 ' touchscreen 1eg0 .052.01**

- True label: NO MATCH | Correct: **Multi-Agent**
- Single LLM: MATCH (conf 0.85): Both listings reference a TomTom XL 330 with a 4.3" touchscreen and similar model numbers (1eg005200 vs garbled 1eg0.052.01), indicating they are the same product despite wording differences.
- Multi-Agent: NO MATCH (conf 0.78): Although both listings share the brand TomTom and the XL 330 series identifier, the product names indicate different variants: Product A is the 'One XL 330' car navigation system, while Product B is t...

**18. blueant black bluetooth headset z9i vs blueant bluetooth hs dual mic nic 091004**

- True label: NO MATCH | Correct: **Multi-Agent**
- Single LLM: MATCH (conf 0.7): Both are BlueAnt Bluetooth headsets with dual microphones, similar price, and overlapping branding; although model identifiers differ, the lack of distinct specs suggests they refer to the same produc...
- Multi-Agent: NO MATCH (conf 0.81): Both agents highlight distinct model identifiers (z9i vs nic 091004) and differing name tokens despite identical brand and price. The lack of overlapping specific identifiers and low similarity scores...

**19. garmin nuvi 205 gps navigation system 0100071740 vs garmin auto nav 010-00715-20**

- True label: NO MATCH | Correct: **Multi-Agent**
- Single LLM: MATCH (conf 0.7): Both listings refer to a Garmin navigation device with very similar part numbers (0100071740 vs 010-00715-20) and the name Nuvi 205 appears in one; the other uses a generic auto nav label but likely t...
- Multi-Agent: NO MATCH (conf 0.92): Both agents correctly note distinct model identifiers and part numbers (Nuvi 205 0100071740 vs AutoNav 010-00715-20). The product lines (Nuvi vs AutoNav) are separate Garmin offerings, and there is no...

**20. speck clear 13 ' macbook see thru hard shell case mb13clrseev2 vs speck products seethru case for apple macbook air mba-clr-see**

- True label: NO MATCH | Correct: **Multi-Agent**
- Single LLM: MATCH (conf 0.85): Both listings describe a Speck clear, see‑through hard plastic case for a 13‑inch MacBook Air, with similar model identifiers (mb13clrseev2 vs mba-clr-see) indicating the same product line; descriptio...
- Multi-Agent: NO MATCH (conf 0.82): Both agents correctly note that the product names reference different MacBook models (13‑inch MacBook vs. MacBook Air). Cases are model‑specific, so they cannot be the same physical product despite si...

**21. sony bud style headphones in silver mdred12lpslv vs bass lover earbuds slver mdred12lp/slv**

- True label: MATCH | Correct: **Single LLM**
- Single LLM: MATCH (conf 0.92): Both listings reference the same model identifier (mdred12lp) and silver color, with similar price range; despite different brand wording, they likely describe the same Sony bud-style headphones.
- Multi-Agent: NO MATCH (conf 0.88): Both agents correctly note that the brand and product type differ (Sony headphones vs Bass Lover earbuds) despite a shared model code fragment. The detailed description for Sony and lack of descriptio...

**22. speck seethru pink hard shell case for 13 ' macbook mb13pnkseev2 vs speck products seethru case for apple macbook air mba-pnk-see**

- True label: NO MATCH | Correct: **Multi-Agent**
- Single LLM: MATCH (conf 0.86): Both listings describe a Speck Seethru pink case for a 13" MacBook Air, with similar model identifiers (mb13pnkseev2 vs mba-pnk-see) and matching color and style, indicating they refer to the same pro...
- Multi-Agent: NO MATCH (conf 0.92): Product A explicitly references the model code 'mb13pnkseev2', which denotes a 13‑inch MacBook (typically a MacBook Pro). Product B references 'mba-pnk-see', indicating a MacBook Air. Speck Seethru ca...

**23. nikon sb-900 af speedlight in black sb900 vs sb-900 af speedlight w / - stnd diffsn dome flt set 4807**

- True label: MATCH | Correct: **Single LLM**
- Single LLM: MATCH (conf 0.92): Both listings reference the Nikon SB-900 AF Speedlight (same model number). The names are essentially identical aside from extra accessory details in B, and the prices are close, indicating they are t...
- Multi-Agent: NO MATCH (conf 0.86): Both listings refer to the Nikon SB‑900 AF speedlight, but Product B explicitly includes a 'standard diffusion dome filter set 4807' accessory, which is not mentioned in Product A. The presence of thi...

**24. canon cyan ink tank cyan cli8c vs canon cli-8c ink cartridge 0621b002**

- True label: MATCH | Correct: **Single LLM**
- Single LLM: MATCH (conf 0.93): Both listings reference Canon's CLI-8C cyan ink cartridge (named similarly), compatible with the same printer series, and have close price points, indicating they are the same product.
- Multi-Agent: NO MATCH (conf 0.85): Both listings reference the Canon CLI-8C cyan cartridge, but Product A's description explicitly calls it a 'compatible' ink tank, implying a third‑party replacement, while Product B includes the OEM p...

**25. sanus 30 ' 58 ' visionmount flat panel tv black tilting wall mount lt25b1 vs sanus visionmount tilting flat panel tv wall mount mt25-b1**

- True label: NO MATCH | Correct: **Single LLM**
- Single LLM: NO MATCH (conf 0.92): The model numbers differ (lt25b1 vs mt25-b1) and descriptions do not share specific features; the names are similar but refer to distinct Sanus VisionMount models, suggesting different products.
- Multi-Agent: MATCH (conf 0.81): Both listings share the same brand (Sanus), product line (VisionMount), and key attributes (tilting flat‑panel TV wall mount). The only discrepancy is the model code (lt25b1 vs mt25‑b1), which is like...

### Summary of Strengths

| Metric                             | Single LLM better | Multi-Agent better |
| ---------------------------------- | ----------------- | ------------------ |
| Correctly matching (FN avoidance)  | 6                 | 0                  |
| Correctly rejecting (FP avoidance) | 6                 | 13                 |
| **Total wins**                     | **12**            | **13**             |

## 6. Architecture Strengths and Weaknesses

### Head-to-Head Comparison

| Metric    | Single LLM | Multi-Agent | Advantage |
| --------- | ---------- | ----------- | --------- |
| Precision | 0.8571     | 0.8807      | Multi     |
| Recall    | 0.9612     | 0.9320      | Single    |
| F1        | 0.9062     | 0.9057      | Single    |
| FP Count  | 33         | 26          | Multi     |
| FN Count  | 8          | 14          | Single    |

### Confidence Distribution

**Single LLM:**

- Average confidence (all): 0.955
- Average confidence (correct): 0.958
- Average confidence (incorrect): 0.812
- Calibration gap: 0.145

**Multi-Agent:**

- Average confidence (all): 0.926
- Average confidence (correct): 0.928
- Average confidence (incorrect): 0.814
- Calibration gap: 0.114

### Token Usage

- **Single LLM:** avg 657 tokens/pair, total 1,257,857 tokens
- **Multi-Agent:** avg 5699 tokens/pair, total 10,918,372 tokens
- **Multi-Agent / Single LLM ratio:** 8.7x

### Single LLM Strengths

- Lower token cost per pair
- Higher F1 score (0.9062 vs 0.9057)
- Better recall (0.9612 vs 0.9320)
- Simpler architecture, less latency per pair

### Single LLM Weaknesses

- Lower precision (0.8571 vs 0.8807)
- No structured tool use for verification
- Single point of failure in reasoning

### Multi-Agent Strengths

- Better precision (0.8807 vs 0.8571)
- Structured tool use provides quantitative evidence
- Dual-perspective (syntactic + semantic) catches different error types
- Orchestrator can resolve agent disagreements

### Multi-Agent Weaknesses

- Lower F1 score (0.9057 vs 0.9062)
- Lower recall (0.9320 vs 0.9612)
- Significantly higher token cost per pair
- More complex architecture increases latency
- Agent consensus does not always lead to correct answer

### Why Single LLM May Edge Out Multi-Agent

The single LLM approach benefits from unified reasoning where all available evidence is weighed holistically in a single pass. The multi-agent system, while providing structured verification through dedicated syntactic and semantic tools, introduces potential failure modes: agent disagreements may confuse the orchestrator, the rigid tool-based analysis may miss nuanced contextual clues that a single LLM captures through its broader reasoning, and the orchestrator must synthesize potentially conflicting agent reports. The overhead of coordination does not always translate to better decisions, particularly when the single LLM already has strong general reasoning capabilities for entity matching.
