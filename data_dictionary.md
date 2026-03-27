# HDB Resale Price Prediction — Data Dictionary

## Target Variable
| Column | Type | Description |
|--------|------|-------------|
| `resale_price` | float | Transaction price in SGD (target for prediction) |

## Transaction Features
| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique transaction identifier |
| `Tranc_YearMonth` | str | Transaction date (YYYY-MM) |
| `Tranc_Year` | int | Transaction year (2012–2021) |
| `Tranc_Month` | int | Transaction month (1–12) |

## Flat Characteristics
| Column | Type | Description |
|--------|------|-------------|
| `town` | str | HDB town (26 towns across Singapore) |
| `flat_type` | str | Flat category: 1/2/3/4/5 ROOM, EXECUTIVE, MULTI-GENERATION |
| `flat_model` | str | Architectural model (20 types: Standard, Improved, Model A, etc.) |
| `full_flat_type` | str | Combined flat_type + flat_model (43 unique) |
| `block` | str | HDB block number |
| `street_name` | str | Street name |
| `address` | str | Full address (block + street) |
| `storey_range` | str | Storey range (e.g., "10 TO 12") |
| `mid_storey` | int | Midpoint of storey range |
| `lower` | int | Lower bound of storey range |
| `upper` | int | Upper bound of storey range |
| `mid` | int | Alternative midpoint |
| `floor_area_sqm` | float | Floor area in square metres |
| `floor_area_sqft` | float | Floor area in square feet (derived) |
| `lease_commence_date` | int | Year lease commenced |
| `hdb_age` | int | Age of flat at transaction |
| `max_floor_lvl` | int | Maximum floor level of the block |
| `year_completed` | int | Year construction completed |

## Building Attributes
| Column | Type | Description |
|--------|------|-------------|
| `residential` | str | Residential use (Y/N) — all Y in dataset |
| `commercial` | str | Commercial component (Y/N) |
| `market_hawker` | str | Market/hawker within building (Y/N) |
| `multistorey_carpark` | str | Multi-storey carpark (Y/N) |
| `precinct_pavilion` | str | Precinct pavilion (Y/N) |
| `total_dwelling_units` | int | Total units in the block |

## Unit Mix (Block-level)
| Column | Type | Description |
|--------|------|-------------|
| `1room_sold` – `multigen_sold` | int | Count of each flat type sold in block |
| `studio_apartment_sold` | int | Studio apartments sold |
| `1room_rental` – `other_room_rental` | int | Rental units by type |

## Location
| Column | Type | Description |
|--------|------|-------------|
| `postal` | str | Postal code |
| `Latitude` | float | Latitude coordinate |
| `Longitude` | float | Longitude coordinate |
| `planning_area` | str | URA planning area (32 areas) |

## Amenity Proximity — Malls
| Column | Type | Description |
|--------|------|-------------|
| `Mall_Nearest_Distance` | float | Distance to nearest mall (metres) |
| `Mall_Within_500m` | float | Number of malls within 500m (blank = 0) |
| `Mall_Within_1km` | float | Number of malls within 1km (blank = 0) |
| `Mall_Within_2km` | float | Number of malls within 2km (blank = 0) |

## Amenity Proximity — Hawker Centres
| Column | Type | Description |
|--------|------|-------------|
| `Hawker_Nearest_Distance` | float | Distance to nearest hawker centre (metres) |
| `Hawker_Within_500m` | float | Hawker centres within 500m (blank = 0) |
| `Hawker_Within_1km` | float | Hawker centres within 1km (blank = 0) |
| `Hawker_Within_2km` | float | Hawker centres within 2km (blank = 0) |
| `hawker_food_stalls` | int | Food stalls in nearest hawker |
| `hawker_market_stalls` | int | Market stalls in nearest hawker |

## Transport
| Column | Type | Description |
|--------|------|-------------|
| `mrt_nearest_distance` | float | Distance to nearest MRT station (metres) |
| `mrt_name` | str | Name of nearest MRT station |
| `bus_interchange` | int | Is nearest MRT a bus interchange? (0/1) |
| `mrt_interchange` | int | Is nearest MRT an interchange station? (0/1) |
| `mrt_latitude` | float | Latitude of nearest MRT |
| `mrt_longitude` | float | Longitude of nearest MRT |
| `bus_stop_nearest_distance` | float | Distance to nearest bus stop (metres) |
| `bus_stop_name` | str | Name of nearest bus stop |
| `bus_stop_latitude` | float | Latitude of nearest bus stop |
| `bus_stop_longitude` | float | Longitude of nearest bus stop |

## Schools
| Column | Type | Description |
|--------|------|-------------|
| `pri_sch_nearest_distance` | float | Distance to nearest primary school (metres) |
| `pri_sch_name` | str | Name of nearest primary school |
| `vacancy` | int | Primary school vacancy count |
| `pri_sch_affiliation` | int | Is primary school affiliated? (0/1) |
| `pri_sch_latitude` | float | Latitude of nearest primary school |
| `pri_sch_longitude` | float | Longitude of nearest primary school |
| `sec_sch_nearest_dist` | float | Distance to nearest secondary school (metres) |
| `sec_sch_name` | str | Name of nearest secondary school |
| `cutoff_point` | int | Secondary school PSLE cutoff point |
| `affiliation` | int | Is secondary school affiliated? (0/1) |
| `sec_sch_latitude` | float | Latitude of nearest secondary school |
| `sec_sch_longitude` | float | Longitude of nearest secondary school |

## Derived Columns (price_per_sqft)
| Column | Type | Description |
|--------|------|-------------|
| `price_per_sqft` | float | **LEAKAGE** — derived from target, excluded from modelling |

## Engineered Features (created in notebook)
| Column | Type | Description |
|--------|------|-------------|
| `remaining_lease` | int | 99 - hdb_age (years of lease remaining) |
| `floor_area_x_storey` | float | floor_area_sqm × mid_storey (interaction) |
| `is_mature_estate` | int | 1 if planning_area is a mature estate, 0 otherwise |
| `mrt_accessibility` | float | Composite MRT score: 1/distance × (1 + interchange) |
| `amenity_density` | float | Mall_Within_2km + Hawker_Within_2km |
| `log_mrt_dist` | float | log1p(mrt_nearest_distance) |
| `log_mall_dist` | float | log1p(Mall_Nearest_Distance) |

---
*Note: "blank = 0" means missing values in amenity count columns represent zero amenities within that radius, not missing data.*
