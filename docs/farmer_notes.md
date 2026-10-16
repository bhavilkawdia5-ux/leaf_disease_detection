# Farmer and KVK conversations (Phase 1)

Goal: check the draft in `ml/classes.yaml` with 5 to 10 farmers or a KVK officer.
Fill one section per person. Short notes are fine.

## Questions to ask

1. Which crops do you grow? Which one loses the most money to leaf disease?
2. Which leaf diseases are most common? In which season?
3. How do you find out which disease it is today? (yourself, shop, KVK, WhatsApp)
4. Show the disease list for their crop: which are common here, which are rare,
   which are missing?
5. Organic or chemical treatment? Who tells you which product and dose?
6. Which phone do you use? Is there mobile internet in your field?
7. Which language do you prefer: Marathi, Hindi or English? Would voice help?
8. Can we photograph your leaves (healthy and diseased) for the dataset?

## Conversations

### 1. Name / village / date

- Crops:
- Common diseases:
- Missing from our list:
- Rare (can drop):
- How they diagnose today:
- Treatment preference:
- Phone and internet:
- Language:
- Photos allowed: yes / no

### 2. Name / village / date

- Crops:
- Common diseases:
- Missing from our list:
- Rare (can drop):
- How they diagnose today:
- Treatment preference:
- Phone and internet:
- Language:
- Photos allowed: yes / no

## Desk review (interim, 2026-10-16)

Farmers could not be contacted in Phase 1, so the draft list was checked
against published sources instead. This is weaker than farmer input: do the
conversations above during Phase 2 field visits, before collecting many own
photos.

| Crop | Class | Finding | Decision |
|---|---|---|---|
| Grape | downy mildew, powdery mildew, anthracnose | NRC Grapes names these the major fungal diseases, causing large losses every year [1] | Keep |
| Grape | bacterial leaf spot | NRC Grapes calls it secondary but growing with climate change [1]; present in the Niphad (Nashik) farm dataset | Keep |
| Grape | rust | Secondary per NRC Grapes [1]; the only Indian report is Karnataka, on *Vitis labrusca* [2]; no public images | **Removed**, re-add if farmers say it is common |
| Onion | purple blotch | Present in all onion areas, more common in kharif [3] | Keep |
| Onion | Stemphylium blight | Of national importance [4]; with anthracnose-twister, the main cause of low kharif yield [5] | Keep |
| Onion | anthracnose (twister) | More severe than purple blotch in a Nashik kharif 2020–21 survey [6] | Keep |
| Onion | downy mildew | Reported mainly from northern hill tracts and humid northern plains [3] | **Removed** |
| Onion | Iris yellow spot virus | First found in India in Jalna and Nashik onion fields [7]; spread by thrips, an emerging problem per ICAR-DOGR [8]; images in COLD | **Added** |
| Onion | thrips damage | Key onion pest, up to 50% yield loss, and the IYSV vector [8] | Keep |
| Tomato | early blight | Major tomato disease in the Konkan region of Maharashtra [9] | Keep |
| Tomato | late blight, leaf curl virus, bacterial spot | Listed for Maharashtra in the government IPM pest list [10]; leaf curl virus is a breeding target at MPKV Rahuri [11] | Keep |
| Tomato | Septoria leaf spot | No Maharashtra-specific source found | Keep, but ask farmers first |

Sources:

1. NRC Grapes (ICAR), package of practices for diseases and insect pests of grapes: https://nrcgrapes.icar.gov.in/sites/default/files/publications/techbul-8-pop-for-disease-insect-pest-on-grapes.pdf
2. First report of *Phakopsora euvitis* leaf rust on grapevine in India (ICRISAT): https://oar.icrisat.org/12499
3. National Horticulture Board, onion diseases: https://www.nhb.gov.in/pdf/vegetable/onion/oni002.pdf
4. Gupta and Gupta (NHRDF Nashik), epidemiology of Stemphylium blight of onion, Indian Journal of Agricultural Sciences 84(9), 2014. Cites Stemphylium blight and purple blotch as diseases of national importance: https://epubs.icar.org.in/index.php/IJAgS/article/download/43461/19223/101963
5. DOGROF-3 organic formulation trials, ICAR-DOGR (Frontiers in Plant Science, 2026): https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1774060/xml
6. Survey of major onion diseases and twister management, Nashik: https://www.researchgate.net/publication/365143849_Survey_Occurrence_of_Major_Onion_Diseases_and_Management_of_Twister_Disease_of_Onion_Allium_cepa_L
7. Iris yellow spot virus in onions, new record from India: https://ndrs.org.uk/article.php?id=011021
8. Karuppaiah and Soumia (ICAR-DOGR), onion thrips and IYSV: https://mail.i-scholar.in/index.php/Rk/article/view/186969
9. Early blight of tomato, Konkan region (Shodhganga thesis): https://shodhganga.inflibnet.ac.in:8443/jspui/handle/10603/120565
10. Vikaspedia, IPM strategies for tomato: https://vikaspedia.in/agriculture/crop-production/integrated-pest-managment/ipm-for-vegetables/ipm-strategies-for-tomato/tomato-pests
11. MPKV Rahuri research report: https://mpkv.ac.in/Uploads/Research/20260819_123737_3fbf99aa75c9858af9fe7ad9fbe7faaf.pdf

## Summary (fill after all conversations)

Interim answers from the desk review; confirm with farmers.

- Final crops: tomato, onion, grape (provisional)
- Diseases added: onion Iris yellow spot virus
- Diseases removed: onion downy mildew, grape rust
- App name: KrishiCare
