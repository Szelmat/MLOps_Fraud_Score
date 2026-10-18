from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.config import JsonDict

TransactionType = Literal["ATM Withdrawal", "Bank Transfer", "Online", "POS"]
DeviceType = Literal["Laptop", "Mobile", "Tablet"]
Location = Literal["London", "Mumbai", "New York", "Sydney", "Tokyo"]
MerchantCategory = Literal["Clothing", "Electronics", "Groceries", "Restaurants", "Travel"]
CardType = Literal["Amex", "Discover", "Mastercard", "Visa"]
AuthenticationMethod = Literal["Biometric", "OTP", "PIN", "Password"]

# Maps API field names to the column names the model was trained on.
FEATURE_COLUMNS = {
    "transaction_amount": "Transaction_Amount",
    "transaction_type": "Transaction_Type",
    "timestamp": "Timestamp",
    "account_balance": "Account_Balance",
    "device_type": "Device_Type",
    "location": "Location",
    "merchant_category": "Merchant_Category",
    "ip_address_flag": "IP_Address_Flag",
    "previous_fraudulent_activity": "Previous_Fraudulent_Activity",
    "daily_transaction_count": "Daily_Transaction_Count",
    "avg_transaction_amount_7d": "Avg_Transaction_Amount_7d",
    "failed_transaction_count_7d": "Failed_Transaction_Count_7d",
    "card_type": "Card_Type",
    "card_age": "Card_Age",
    "transaction_distance": "Transaction_Distance",
    "authentication_method": "Authentication_Method",
    "risk_score": "Risk_Score",
    "is_weekend": "Is_Weekend",
}

EXAMPLE_TRANSACTION: JsonDict = {
    "transaction_amount": 39.85,
    "transaction_type": "POS",
    "timestamp": "2023-01-22T23:47:00",
    "account_balance": 33069.86,
    "device_type": "Tablet",
    "location": "Mumbai",
    "merchant_category": "Travel",
    "ip_address_flag": True,
    "previous_fraudulent_activity": False,
    "daily_transaction_count": 2,
    "avg_transaction_amount_7d": 81.6,
    "failed_transaction_count_7d": 2,
    "card_type": "Amex",
    "card_age": 223,
    "transaction_distance": 617.12,
    "authentication_method": "PIN",
    "risk_score": 0.6742,
    "is_weekend": True,
}


class Transaction(BaseModel):
    """A card transaction to score."""

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"examples": [EXAMPLE_TRANSACTION]}
    )

    transaction_amount: float = Field(ge=0)
    transaction_type: TransactionType
    timestamp: datetime
    account_balance: float
    device_type: DeviceType
    location: Location
    merchant_category: MerchantCategory
    ip_address_flag: bool
    previous_fraudulent_activity: bool
    daily_transaction_count: int = Field(ge=0)
    avg_transaction_amount_7d: float = Field(ge=0)
    failed_transaction_count_7d: int = Field(ge=0)
    card_type: CardType
    card_age: int = Field(ge=0, description="Card age in days")
    transaction_distance: float = Field(ge=0)
    authentication_method: AuthenticationMethod
    risk_score: float = Field(ge=0, le=1)
    is_weekend: bool

    def to_feature_row(self) -> dict[str, Any]:
        """Returns the transaction in the raw dataset's column names and formats."""
        row = {FEATURE_COLUMNS[name]: value for name, value in self.model_dump().items()}
        row["Timestamp"] = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        for column in ("IP_Address_Flag", "Previous_Fraudulent_Activity", "Is_Weekend"):
            row[column] = int(row[column])
        return row


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    fraud_probability: float = Field(ge=0, le=1)
    is_fraud: bool
    threshold: float


class BatchPrediction(BaseModel):
    predictions: list[Prediction] = Field(
        description="One prediction per transaction, in request order"
    )


class Health(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
