"""app/main.py — FastAPI prediction service for Egyptian Auction Price Predictor."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Egyptian Auction Price Predictor",
    description="""
Predicts the **final selling price (EGP)** for items listed on Egyptian online auctions,
plus an **80% prediction interval** to guide reserve and buy-now pricing.

## What the seller must provide
| Field | Why it's needed |
|-------|----------------|
| `item_title` | Processed by TF-IDF — captures product-specific keywords |
| `item_description` | Processed by TF-IDF — adds detail about condition, accessories, etc. |
| 13 structured fields | Category, price, seller info, listing timing |

## Model
Ensemble: Random Forest + LightGBM + XGBoost (log-target blend)
Features: 100 TF-IDF (title+desc) + structured interactions

## How to use the price range
- `price_range_low_egp`  → set your **reserve price** at or just below this value
- `price_range_high_egp` → set your **buy-now price** at or just above this value
- `predicted_final_selling_price_egp` → single best estimate of the final sale price
    """,
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AuctionListing(BaseModel):
    # ── Text fields (processed via TF-IDF, not dropped) ───────────────────────
    item_title: str = Field(
        ...,
        example="Apple MacBook Pro 14-inch M3 Pro 18GB 512GB Space Black",
        description=(
            "Full title of the item you want to list. "
            "Be specific — include brand, model, specs, and condition keywords. "
            "This is processed by TF-IDF and directly affects the predicted price."
        ),
    )
    item_description: str = Field(
        ...,
        example=(
            "MacBook Pro 14-inch with M3 Pro chip, 18GB unified memory, 512GB SSD. "
            "Purchased 12 months ago. No scratches, original box and accessories included. "
            "Battery health 94%. AppleCare valid until 2026."
        ),
        description=(
            "Detailed description of the item. Include condition details, accessories, "
            "purchase history, and any defects. "
            "This is processed by TF-IDF and directly affects the predicted price."
        ),
    )
    # ── Structured features ───────────────────────────────────────────────────
    category: str = Field(
        ..., example="Electronics",
        description="Main category. Options: Electronics, Fashion, Home & Garden, Sports, Vehicles, Books, Collectibles, Other",
    )
    subcategory: str = Field(
        ..., example="Laptops",
        description="Sub-category (e.g. Laptops, Smartphones, Watches, Sneakers)",
    )
    brand: str = Field(
        ..., example="Apple",
        description="Brand name (e.g. Apple, Samsung, Sony, Generic)",
    )
    condition: str = Field(
        ..., example="Like New",
        description="Physical condition. Options: For Parts | Poor | Acceptable | Fair | Good | Very Good | Excellent | Like New | New",
    )
    product_age: int = Field(
        ..., ge=0, le=240, example=12,
        description="Age of product in months. 0 = brand new.",
    )
    starting_price: float = Field(
        ..., gt=0, example=25000.0,
        description="Starting bid price in EGP. Must be positive.",
    )
    auction_duration: int = Field(
        ..., ge=1, le=30, example=7,
        description="Auction duration in days (1–30).",
    )
    listing_day_of_week: str = Field(
        ..., example="Saturday",
        description="Day the auction is listed. Options: Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday",
    )
    listing_hour: int = Field(
        ..., ge=0, le=23, example=20,
        description="Hour of listing (24h format). 0=midnight, 12=noon, 20=8pm.",
    )
    seller_rating: float = Field(
        ..., ge=0.0, le=5.0, example=4.8,
        description="Seller rating (0.0 = worst, 5.0 = best).",
    )
    seller_total_sales: int = Field(
        ..., ge=0, example=120,
        description="Total completed sales by this seller.",
    )
    seller_account_age: int = Field(
        ..., ge=0, example=36,
        description="Seller account age in months.",
    )
    verified_seller: int = Field(
        ..., ge=0, le=1, example=1,
        description="1 = verified seller, 0 = not verified.",
    )


class PredictionResponse(BaseModel):
    item_title: str
    predicted_final_selling_price_egp: float
    price_range_low_egp: float
    price_range_high_egp: float
    currency: str = "EGP"
    model: str = "RF + LightGBM + XGBoost Ensemble (TF-IDF + Structured)"
    model_version: str = "1.2.0"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Egyptian Auction Price Predictor",
        "version": "1.2.0",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "POST /predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    try:
        from scripts.predict import load_pipeline
        *_, bundle, _ = load_pipeline()
        return HealthResponse(status="ok", model_loaded=True, model_type=bundle["type"])
    except Exception as e:
        return HealthResponse(status=f"error: {e}", model_loaded=False, model_type="unknown")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(listing: AuctionListing):
    """
    Predict final selling price + 80% price range for an auction listing.

    Both `item_title` and `item_description` are **required** and affect the prediction
    through TF-IDF feature extraction — they are not just for display.

    ### Response
    | Field | Description |
    |-------|-------------|
    | `predicted_final_selling_price_egp` | Best single price estimate |
    | `price_range_low_egp`  | Lower bound of 80% interval — use as reserve price floor |
    | `price_range_high_egp` | Upper bound of 80% interval — use as buy-now price ceiling |
    """
    try:
        from scripts.predict import predict_single_with_interval
        data   = listing.model_dump()
        result = predict_single_with_interval(data)   # title+desc passed through
        return PredictionResponse(
            item_title                          = listing.item_title,
            predicted_final_selling_price_egp   = result["predicted_price"],
            price_range_low_egp                 = result["price_range_low"],
            price_range_high_egp                = result["price_range_high"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")