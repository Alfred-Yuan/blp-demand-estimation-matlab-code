/*
Simple logit using stata
Authors: Alfred Yuan
Date: August 18, 2026
*/

clear all
set more off

***********************************
* Part A: Load and clean the data *
***********************************

/*
Create simple-logit estimation data in Stata.
Run this do-file from:
Cleaned Project/code/main/codes/demand_estimation
*/
global code_dir "`c(pwd)'"
global repo_dir "$code_dir/../.."
global data_dir "/Users/alfredyuan/Dropbox/GS_Network/Data/final_estimation_data/demand"

global household_file "$data_dir/Household Size.xlsx"

*Step A1. Load main product-market data.

/* 
Here, I use python to transform data to dta version, if you
have the most updated stata, you can omit this line and use:
import parquet using "$data_dir/df.parquet", clear
*/
use "$data_dir/df.dta", clear


* Step A2. Variable-name mapping.
rename Year year
rename Geo_Market_id location_id
rename manufacturer_id firm_id
rename Quantity sales
rename price price_real
rename log_price ln_price
rename log_size ln_volume
rename log_ev_range ln_driving_range
rename log_power ln_hp
rename ev EV
rename gv GV
rename is_bev BEV
rename is_phev PHEV
rename is_reev REEV
rename is_ice ICE
rename fueltype_num fuel_type_id
rename sizesegment_num segment_id
rename bodytype_num bodytype_id
rename Population population
rename duration duration_yr
rename CentralS centralS_real
rename LocalS localS_real

rename z_ev_iv_charger_poi_cs_othermkt z_ev_cs_othermkt
rename z_ev_iv_charger_poi_cs_cost z_ev_cs_cost
rename z_gv_iv_charger_carfix_poi_gs_ot z_gv_carfix_gs_other
rename z_gv_iv_charger_bus_terminal_poi z_gv_bus_gs_other
rename z_gv_iv_charger_car_service_poi_ z_gv_service_gs_other
rename z_gv_iv_charger_road_poi_oil_pri z_gv_road_oil_price
rename z_gv_iv_charger_carfix_poi_oil_p z_gv_carfix_oil_price
rename _0z_gv_iv_charger_bus_terminal_p z_gv_bus_oil_price
rename _1z_gv_iv_charger_car_service_po z_gv_service_oil_price
rename z_gv_iv_charger_road_poi_gs_othe z_gv_road_gs_other

sort year location_id model_id

*****************************************
* Part B: Construct necessary variables *
*****************************************

* Step B0: drop low sales models
drop if sales <= 30

* Step B1. Merge household size.

preserve
    import excel using "$household_file", sheet("Household Size") firstrow clear
    rename Year year
    keep year Geo_Market total_hh_size
    duplicates drop
    tempfile household_size
    save `household_size', replace
restore

* Merge the household size data with our main sales data
merge m:1 year Geo_Market using `household_size', keep(match) nogen
sort year location_id model_id


* Step B2. Construct market size, product share, outside share, and mean utility.

gen double mkt_size = population * 10000 / (total_hh_size * 5) // Set the replacement cycle as 5 years

gen double s_jt = sales / mkt_size //Generate market share
bysort market_id: egen double total_share = total(s_jt)
gen double s_0 = 1 - total_share

keep if s_0 > 1e-4 & s_jt > 0 //Drop those negative outside option

gen double mean_utility = ln(s_jt) - ln(s_0)


* Step B3. Rescale attributes and create network-density controls.

replace ln_volume = ln_volume - ln(1000000) // Change the unit to m^3
replace fuel_cost_real = fuel_cost_real / 100 // Change the unit to 100yuan

gen double cs_density = cs_count / population * 100 //number / million people
gen double gs_density = gs_count / population * 100
gen double ln_cs_density_ev = EV * ln(cs_density + 1)
gen double ln_gs_density_gv = GV * ln(gs_density + 1)

**********************************************
* Part C: Construct differentiation-style IV *
**********************************************

* Step C0. Setup Basic Specification.

global endovars price_real ln_cs_density_ev ln_gs_density_gv 
global price_iv st_rate centralS_real
global cs_iv z_ev_cs_othermkt z_ev_cs_cost
global gs_iv z_gv_carfix_gs_other z_gv_bus_gs_other z_gv_service_gs_other ///
    z_gv_road_oil_price z_gv_carfix_oil_price z_gv_bus_oil_price ///
    z_gv_service_oil_price z_gv_road_gs_other
global excluded_iv $price_iv $cs_iv $gs_iv
global controls ln_volume ln_driving_range ln_hp fuel_cost_real
global FE brand_model_id location_id year fuel_type_id duration_yr bodytype_id segment_id

* Step C1. Program for p_hat-based differentiation IV.
capture program drop make_phat_iv
program define make_phat_iv
	* Parse selected controls, excluded IVs, FEs, bandwidth, and IV type.
	syntax, Controls(string) IVS(string) FES(string) ///
		[Band(real 0.75) Type(string)]

	* Use local-count IV by default.
	if "`type'" == "" local type "lc"

	* Remove old output and temporary variables before rebuilding p_hat IVs.
	foreach v in p_hat z_p_hat lc_iv_p_hat ed_iv_p_hat N_close z2_p_hat sum_z sum_z2 N_market {
		capture drop `v'
	}

	* First stage: predict the exogenous component of price under the selected spec.
	quietly reghdfe price_real `controls' `ivs', ///
		absorb(`fes') resid keepsingletons
	predict double p_hat if e(sample), xbd

	* Standardize predicted price for distance/count calculations.
	egen double z_p_hat = std(p_hat)

	* Local-count IV: number of products within the chosen standardized bandwidth.
	if inlist("`type'", "lc", "both") {
		qui rangestat (count) N_close = model_id, ///
			interval(z_p_hat -`band' `band') by(market_id)
		gen double lc_iv_p_hat = N_close - 1
		drop N_close
	}

	* Euclidean-distance IV: distance from all other products in the same market.
	if inlist("`type'", "ed", "both") {
		gen double z2_p_hat = z_p_hat^2
		bysort market_id: egen double sum_z = total(z_p_hat)
		bysort market_id: egen double sum_z2 = total(z2_p_hat)
		bysort market_id: egen long N_market = count(z_p_hat)
		gen double ed_iv_p_hat = sqrt(max(0, ///
			sum_z2 - 2*z_p_hat*sum_z + N_market*z_p_hat^2))
		drop z2_p_hat sum_z sum_z2 N_market
	}
end


* Step C2. Program for characteristic-based differentiation IVs.
capture program drop make_char_iv
program define make_char_iv
	* Parse the characteristic list, bandwidth, and IV type.
	syntax, Vars(string) [Band(real 0.75) Type(string)]

	* Use local-count IV by default.
	if "`type'" == "" local type "lc"

	* Build differentiation IVs separately for each selected characteristic.
	foreach x of local vars {
		* Remove old outputs and temporary variables for this characteristic.
		capture drop z_`x' lc_iv_`x' ed_iv_`x'
		capture drop N_close z2_`x' sum_z sum_z2 N_market

		* Standardize the characteristic across products.
		egen double z_`x' = std(`x')

		* Local-count IV: number of nearby products in the same market.
		if inlist("`type'", "lc", "both") {
			qui rangestat (count) N_close = model_id, ///
				interval(z_`x' -`band' `band') by(market_id)
			gen double lc_iv_`x' = N_close - 1
			drop N_close
		}

		* Euclidean-distance IV: within-market distance from other products.
		if inlist("`type'", "ed", "both") {
			gen double z2_`x' = z_`x'^2
			bysort market_id: egen double sum_z = total(z_`x')
			bysort market_id: egen double sum_z2 = total(z2_`x')
			bysort market_id: egen long N_market = count(z_`x')
			gen double ed_iv_`x' = sqrt(max(0, ///
				sum_z2 - 2*z_`x'*sum_z + N_market*z_`x'^2))
			drop z2_`x' sum_z sum_z2 N_market
		}
	}
end

* Step C3. Program for EV driving-range differentiation IV.
capture program drop make_range_iv
program define make_range_iv
	* Parse bandwidth and IV type.
	syntax, [Band(real 0.75) Type(string)]

	* Use local-count IV by default.
	if "`type'" == "" local type "lc"

	* Remove old range IVs and temporary variables.
	capture drop z_range lc_iv_ln_driving_range ed_iv_ln_driving_range N_close z2_range sum_z_range sum_z2_range N_range

	* Use temporary variables so repeated calls do not leave z_range in memory.
	tempvar z_range n_close z2_range sum_z_range sum_z2_range n_range

	* Standardize driving range using EV observations only.
	egen double `z_range' = std(ln_driving_range) if EV == 1

	* Local-count IV for EV range; set GV observations to zero.
	if inlist("`type'", "lc", "both") {
		qui rangestat (count) `n_close' = model_id, ///
			interval(`z_range' -`band' `band') by(market_id)
		gen double lc_iv_ln_driving_range = `n_close' - 1 if EV == 1
		replace lc_iv_ln_driving_range = 0 if GV == 1
	}

	* Euclidean-distance IV for EV range; set GV observations to zero.
	if inlist("`type'", "ed", "both") {
		gen double `z2_range' = `z_range'^2
		bysort market_id: egen double `sum_z_range' = total(`z_range')
		bysort market_id: egen double `sum_z2_range' = total(`z2_range')
		bysort market_id: egen long `n_range' = count(`z_range')
		gen double ed_iv_ln_driving_range = sqrt(max(0, ///
			`sum_z2_range' - 2*`z_range'*`sum_z_range' + `n_range'*`z_range'^2)) if EV == 1
		replace ed_iv_ln_driving_range = 0 if GV == 1
	}
end

* Step C4. Program for GV differentiation IV.
capture program drop make_gv_iv
program define make_gv_iv
	* Parse the variable, bandwidth, and IV type.
	syntax, [Var(name) Band(real 0.75) Type(string)]

	* Use local-count IV by default.
	if "`type'" == "" local type "lc"
	if "`var'" == "" local var "fuel_cost_real"

	* Remove old GV IVs and temporary variables.
	capture drop z_fuel_cost_real lc_iv_gv ed_iv_gv N_close_gv z2_gv sum_z_gv sum_z2_gv N_gv

	* Use temporary variables so repeated calls do not leave standardized values in memory.
	tempvar z_gv n_close z2_gv sum_z_gv sum_z2_gv n_gv

	* Standardize the selected variable for GV-style differentiation.
	egen double `z_gv' = std(`var')

	* Local-count IV within market and GV status.
	if inlist("`type'", "lc", "both") {
		qui rangestat (count) `n_close' = model_id, ///
			interval(`z_gv' -`band' `band') by(market_id GV)
		capture drop lc_iv_gv
		gen double lc_iv_gv = `n_close' - 1 if !missing(`z_gv')
	}

	* Euclidean-distance IV within market and GV status.
	if inlist("`type'", "ed", "both") {
		gen double `z2_gv' = `z_gv'^2
		bysort market_id GV: egen double `sum_z_gv' = total(`z_gv')
		bysort market_id GV: egen double `sum_z2_gv' = total(`z2_gv')
		bysort market_id GV: egen long `n_gv' = count(`z_gv')
		capture drop ed_iv_gv
		gen double ed_iv_gv = sqrt(max(0, ///
			`sum_z2_gv' - 2*`z_gv'*`sum_z_gv' + `n_gv'*`z_gv'^2))
	}
end


* Step C5. Count IV for GV model variety.
capture drop count_iv_gv tag_gv_model

* Tag each distinct GV model once within each market.
egen byte tag_gv_model = tag(market_id model_id) if GV == 1 & !missing(model_id)

* Count distinct GV models in each market and assign the count to all products.
bysort market_id: egen long count_iv_gv = total(tag_gv_model)

drop tag_gv_model


**********************************************************
* Part D: Try simple logit with different sprcifications *
**********************************************************

/*
Spec 1: Default specification 
(price iv: st_rate centralS_real
 FEs: brand_model_id location_id year fuel_type_id duration_yr
 cs and gs density uses all their ivs
 controls: ln_volume ln_driving_range ln_hp fuel_cost_real)
*/

**replace price_real = ln(price_real)

reghdfe mean_utility $endovars $controls , a($FE) cluster(market_id)

reghdfe mean_utility $endovars $controls , a($FE) cluster(market_id)
ivreghdfe mean_utility $controls ( $endovars = $excluded_iv ct_rate ), a($FE) cluster(market_id)


make_phat_iv, controls("$controls") ivs("$excluded_iv") fes("$FE") band(0.75) type(lc)
make_gv_iv, var(p_hat) band(0.75) type(lc)
ivreghdfe mean_utility $controls ( $endovars = $excluded_iv count_iv_gv lc_iv_p_hat), a($FE) cluster(market_id)


/*
Spec 2: Change price iv: st_rate centralS_real ct_rate
*/

ivreghdfe mean_utility $controls Displacement ( $endovars = $excluded_iv ct_rate ), a($FE) cluster(market_id) first

make_phat_iv, controls("$controls") ivs("$excluded_iv ct_rate") fes("$FE") band(0.75) type(lc)
ivreghdfe mean_utility $controls ( $endovars = $excluded_iv count_iv_gv lc_iv_p_hat), a($FE) cluster(market_id)


/*
Spec 3: Change price iv: centralS_real only
*/

ivreghdfe mean_utility $controls ( $endovars = centralS_real $cs_iv $gs_iv ), a($FE) cluster(market_id)

make_phat_iv, controls("$controls") ivs("centralS_real $cs_iv $gs_iv") fes("$FE") band(0.75) type(lc)
ivreghdfe mean_utility $controls ( $endovars = centralS_real $cs_iv $gs_iv  lc_iv_gv lc_iv_p_hat), a($FE) cluster(market_id)


/*
Spec 4: Change price iv: st_rate only
*/

ivreghdfe mean_utility $controls ( $endovars = st_rate $cs_iv $gs_iv ), a($FE) cluster(market_id)

make_phat_iv, controls("$controls") ivs("st_rate $cs_iv $gs_iv") fes("$FE") band(0.75) type(lc)
ivreghdfe mean_utility $controls ( $endovars = st_rate $cs_iv $gs_iv  lc_iv_gv lc_iv_p_hat), a($FE) cluster(market_id)


/*
Spec 5: Change price iv: ct_rate only
*/

ivreghdfe mean_utility $controls ( $endovars = ct_rate $cs_iv $gs_iv ), a($FE) cluster(market_id)

make_phat_iv, controls("$controls") ivs("ct_rate $cs_iv $gs_iv") fes("$FE") band(0.75) type(lc)
ivreghdfe mean_utility $controls ( $endovars = ct_rate $cs_iv $gs_iv  lc_iv_gv lc_iv_p_hat), a($FE) cluster(market_id)


/*
Spec 6: Change price iv: centralS_real + ct_rate
*/

ivreghdfe mean_utility $controls ( $endovars = centralS_real ct_rate $cs_iv $gs_iv ), a($FE) cluster(market_id)

make_phat_iv, controls("$controls") ivs("centralS_real ct_rate $cs_iv $gs_iv") fes("$FE") band(0.75) type(lc)
ivreghdfe mean_utility $controls ( $endovars = centralS_real ct_rate $cs_iv $gs_iv  lc_iv_gv lc_iv_p_hat), a($FE) cluster(market_id)

/*
Spec 7: Change price iv: st_rate + ct_rate
*/

ivreghdfe mean_utility $controls ( $endovars = st_rate ct_rate $cs_iv $gs_iv ), a($FE) cluster(market_id)

make_phat_iv, controls("$controls") ivs("st_rate ct_rate $cs_iv $gs_iv") fes("$FE") band(0.75) type(lc)
ivreghdfe mean_utility $controls ( $endovars = st_rate ct_rate $cs_iv $gs_iv  lc_iv_gv lc_iv_p_hat), a($FE) cluster(market_id)


/*
Spec 8: Change price iv: total_tax
*/
ivreghdfe mean_utility $controls ( $endovars = total_tax $cs_iv $gs_iv ), a($FE) cluster(market_id)

make_phat_iv, controls("$controls") ivs("total_tax $cs_iv $gs_iv") fes("$FE") band(0.75) type(lc)
ivreghdfe mean_utility $controls ( $endovars = total_tax $cs_iv $gs_iv lc_iv_gv lc_iv_p_hat), a($FE) cluster(market_id)

/*
Spec 9: Change price iv: total_tax + centralS_real
*/
ivreghdfe mean_utility $controls ( $endovars = total_tax centralS_real $cs_iv $gs_iv ), a($FE) cluster(market_id)

make_phat_iv, controls("$controls") ivs("total_tax centralS_real $cs_iv $gs_iv") fes("$FE") band(0.75) type(lc)
ivreghdfe mean_utility $controls ( $endovars = total_tax centralS_real $cs_iv $gs_iv lc_iv_gv lc_iv_p_hat), a($FE) cluster(market_id)

/*
Spec 11: Change FEs: add bodytype_id and segment_id; change price iv: add ct_rate
*/

ivreghdfe mean_utility $controls ( $endovars = $excluded_iv ct_rate ), a($FE bodytype_id segment_id) cluster(market_id)

make_phat_iv, controls("$controls") ivs("$excluded_iv ct_rate") fes("$FE bodytype_id segment_id") band(0.75) type(lc)
ivreghdfe mean_utility $controls ( $endovars = $excluded_iv ct_rate lc_iv_gv lc_iv_p_hat), a($FE bodytype_id segment_id) cluster(market_id)


/*
Spec 12: Change differentiation-style iv: band = 1.0
*/

make_phat_iv, controls("$controls") ivs("$excluded_iv") fes("$FE") band(1.0) type(lc)
make_gv_iv, var(fuel_cost_real) band(1.0) type(lc)
ivreghdfe mean_utility $controls ( $endovars = $excluded_iv  lc_iv_gv lc_iv_p_hat), a($FE) cluster(market_id)

/*
Spec 13: Change differentiation-style iv: use ed instead of lc
*/

make_phat_iv, controls("$controls") ivs("$excluded_iv") fes("$FE") band(0.75) type(ed)
make_gv_iv, var(fuel_cost_real) band(0.75) type(ed)
ivreghdfe mean_utility $controls ( $endovars = $excluded_iv  ed_iv_gv ed_iv_p_hat), a($FE) cluster(market_id)

/*
Spec 14: Change differentiation-style iv: add all possible diff ivs
*/

make_phat_iv, controls("$controls") ivs("$excluded_iv") fes("$FE") band(0.75) type(lc)
make_gv_iv, var(fuel_cost_real) band(0.75) type(lc)
make_range_iv, band(0.75) type(lc)
make_char_iv, vars("ln_hp ln_volume fuel_cost_real") band(0.75) type(lc)
ivreghdfe mean_utility $controls ( $endovars = $excluded_iv  lc_iv_*), a($FE) cluster(market_id)


/*
Spec 15: Drop low sales vehicles
*/
preserve
	drop if sales <= 20
	
	reghdfe mean_utility $endovars $controls , a($FE) cluster(market_id)

	ivreghdfe mean_utility $controls ( $endovars = $excluded_iv ), a($FE) cluster(market_id)
	
	make_phat_iv, controls("$controls") ivs("$excluded_iv") fes("$FE") band(0.75) type(lc)
	make_gv_iv, var(fuel_cost_real) band(0.75) type(lc)
	ivreghdfe mean_utility $controls ( $endovars = $excluded_iv lc_iv_gv lc_iv_p_hat ), a($FE) cluster(market_id)
restore

/*
Spec 16: Use alternative diff iv for gv
*/
make_phat_iv, controls("$controls") ivs("$excluded_iv") fes("$FE") band(0.75) type(lc)
make_gv_iv, var(p_hat) band(0.75) type(lc) // Use p_hat instead of fuel_cost_real

ivreghdfe mean_utility $controls ( $endovars = $excluded_iv lc_iv_gv lc_iv_p_hat ), a($FE) cluster(market_id)

ivreghdfe mean_utility $controls ( $endovars = $excluded_iv count_iv_gv lc_iv_p_hat ), a($FE) cluster(market_id)


/*
Spec 17: Use check the coef for gv
*/
reghdfe mean_utility $endovars $controls i.fuel_type_id , a(brand_model_id ///
location_id year duration_yr bodytype_id segment_id) cluster(market_id)

ivreghdfe mean_utility $controls i.fuel_type_id ( $endovars = $excluded_iv ), ///
a(brand_model_id location_id year duration_yr bodytype_id segment_id) cluster(market_id)

make_phat_iv, controls("$controls") ivs("$excluded_iv") fes("$FE") band(0.75) type(lc)
make_gv_iv, var(p_hat) band(0.75) type(lc)
ivreghdfe mean_utility $controls ( $endovars = $excluded_iv count_iv_gv lc_iv_p_hat), a($FE) cluster(market_id)
