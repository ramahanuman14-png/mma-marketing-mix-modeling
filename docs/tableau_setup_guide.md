# Tableau Dashboard Setup Guide
## Marketing Mix Modeling — 4 Sheet Dashboard

---

## FILES TO LOAD (copy both to your Mac first)
- tableau_fact_monthly.csv     → monthly GMV + spend + promo data
- tableau_channel.csv          → channel-level spend + ROAS + optimization

---

## STEP 1 — Connect Data Sources

1. Open Tableau Desktop 2023.3
2. Left panel → Connect → Text File
3. Load: tableau_fact_monthly.csv
4. Click "Add" → load: tableau_channel.csv
5. In the relationship canvas:
   - Drag tableau_channel onto the canvas
   - Link: tableau_fact_monthly[Date] = tableau_channel[Date]
   - Relationship type: Many to Many (both are facts)

---

## STEP 2 — Create Calculated Fields

Go to: Analysis → Create Calculated Field

### GMV in Billions
```
Name: GMV (₹B)
Formula: SUM([Total Gmv M]) / 1000
```

### Media Efficiency
```
Name: Media Efficiency
Formula: SUM([Total Gmv M]) / SUM([Total Spend M])
```

### Promo Flag Label
```
Name: Promo Label
Formula: IF [Has Promo] = 1 THEN "Promo Month" ELSE "Normal Month" END
```

### Budget Change Direction
```
Name: Change Direction
Formula: IF [Change M] > 0 THEN "Increase" ELSE "Decrease" END
```

### Channel ROAS Label
```
Name: ROAS Label
Formula: "₹" + STR(ROUND([Coeff Mean],0)) + "M"
```

### Optimal vs Current
```
Name: Spend Variance M
Formula: [Optimal Spend M] - [Current Spend M]
```

---

## STEP 3 — Build 4 Sheets

---

### SHEET 1: Revenue Trend
**Name it:** "01 Revenue Trend"

```
Rows:    Total_GMV_M (SUM)
Columns: Date (Month)
Marks:   Line
Color:   Promo Label (Blue=Normal, Orange=Promo)
Size:    increase line thickness to 2.5
```

Steps:
1. Drag Date → Columns → right-click → Month (continuous)
2. Drag Total_GMV_M → Rows
3. Drag Promo_Label → Color
4. Change mark type to Line
5. Add reference line: Analytics pane → Average Line
6. Format: remove gridlines, set title "Monthly Revenue Trend"

---

### SHEET 2: Channel Spend vs ROAS
**Name it:** "02 Channel ROAS"

```
Rows:    Channel_Name
Columns: Coeff_Mean (AVG)
Marks:   Bar (horizontal)
Color:   Coeff_Mean (gradient green)
Label:   ROAS Label
Sort:    Descending by Coeff_Mean
```

Steps:
1. Drag Channel_Name → Rows
2. Drag Coeff_Mean → Columns → change to AVG
3. Sort descending
4. Drag Coeff_Mean → Color → Edit colors → Green sequential
5. Drag ROAS_Label → Label
6. Add reference line at AVG
7. Title: "Channel ROAS Coefficient (₹M per saturation unit)"

---

### SHEET 3: Budget Optimization
**Name it:** "03 Budget Optimizer"

```
Type:    Grouped Bar Chart
Rows:    Channel_Name
Columns: Current_Spend_M + Optimal_Spend_M
Color:   Measure Names (grey=Current, blue=Optimal)
```

Steps:
1. Drag Channel_Name → Rows
2. Drag Current_Spend_M → Columns
3. Drag Optimal_Spend_M → Columns (hold Shift)
4. Right-click axis → Dual Axis
5. Mark type → Bar
6. Change_Direction → Color
   - Increase = #34a853 (green)
   - Decrease = #ea4335 (red)
7. Add Change_Pct as label
8. Title: "Current vs Optimal Budget Allocation"

---

### SHEET 4: Spend by Channel Over Time
**Name it:** "04 Spend Breakdown"

```
Type:    Stacked Area Chart
Rows:    Spend_M (SUM)
Columns: Date (Month)
Color:   Channel_Name
Marks:   Area
```

Steps:
1. Use tableau_channel as data source
2. Drag Date → Columns → Month (continuous)
3. Drag Spend_M → Rows
4. Drag Channel_Name → Color
5. Change mark type to Area
6. Assign colors:
   TV           → #1a73e8
   Digital      → #34a853
   Sponsorship  → #fbbc04
   Content Mktg → #ea4335
   Online Mktg  → #9c27b0
   Affiliates   → #00897b
   SEM          → #ff6d00
7. Title: "Monthly Media Spend by Channel"

---

## STEP 4 — Build the Dashboard

1. Click "New Dashboard" (bottom tab)
2. Set size: Fixed → 1400 x 900 px
3. Drag sheets in this layout:

```
┌────────────────────────────────────────────────────────┐
│  MARKETING MIX MODELING DASHBOARD         [MMM Logo]   │
│  Jul 2015 – Jun 2016                                   │
├──────────┬──────────┬──────────┬──────────┬────────────┤
│ Total    │ Total    │ Overall  │ Best     │ Model      │
│ GMV      │ Spend    │ ROAS     │ Channel  │ MAPE       │
│ ₹4.06B   │ ₹7.94B   │ 0.51x    │ SEM      │ 9.4%       │
├──────────┴──────────┴──────────┴──────────┴────────────┤
│                                                        │
│         Sheet 1: Revenue Trend (60% width)             │
│                                                        │
├────────────────────────┬───────────────────────────────┤
│  Sheet 2: ROAS Bars    │  Sheet 4: Spend Breakdown     │
│  (50% width)           │  (50% width)                  │
├────────────────────────┴───────────────────────────────┤
│         Sheet 3: Budget Optimization (full width)      │
└────────────────────────────────────────────────────────┘
```

### KPI Cards (use Text objects):
- Drag "Text" from Objects panel
- Type the KPI value manually for now:
  - Total GMV: ₹4.06B
  - Total Spend: ₹7.94B
  - Overall ROAS: 0.51x
  - Best Channel: SEM (₹130M coeff)
  - Model MAPE: 9.4%

---

## STEP 5 — Formatting

### Dashboard-wide:
- Background: #f8f9fa (light grey)
- Title font: Tableau Bold 18pt
- Remove all outer borders
- Add thin dividers between sections (#e0e0e0)

### Each Sheet:
- Remove sheet title borders
- Font: Tableau Book 10pt for axes
- Font: Tableau Bold 12pt for sheet titles

---

## STEP 6 — Save and Export

```
File → Save As → mmm_dashboard.twb
File → Export → Export as PDF (for sharing without Tableau)
File → Export → Export Packaged Workbook (.twbx)
         ↑ This bundles CSV files too — share this with anyone
```

---

## Interview Talking Points

"I built a 4-sheet Tableau dashboard connected to two CSV data sources
joined on the Date field. I created calculated fields for ROAS efficiency
and budget variance. The dashboard answers three business questions:
where is revenue coming from, which channels deliver best ROI, and
how should the budget be reallocated to maximize revenue."
