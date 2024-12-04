from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from bonito import Bonito, SamplingParams
from datasets import load_dataset
import uvicorn

app = FastAPI(
    title="AutoBonito API",
    description="A FastAPI implementation of AutoBonito for synthetic dataset generation",
    version="1.0.0"
)

# Pydantic models for request/response validation
class SamplingParameters(BaseModel):
    max_tokens: int = Field(default=256, description="Maximum number of tokens to generate")
    top_p: float = Field(default=0.95, description="Top-p sampling parameter")
    temperature: float = Field(default=0.5, description="Temperature for sampling")
    n: int = Field(default=1, description="Number of samples to generate")

class GenerationRequest(BaseModel):
    bonito_model: str = Field(
        default="NousResearch/Genstruct-7B",
        description="Model to use for generation"
    )
    dataset: str = Field(
        description="Dataset name on Huggingface Hub"
    )
    unannotated_text: str = Field(
        description="Path or name of unannotated text file"
    )
    split: str = Field(
        default="train",
        description="Dataset split to use"
    )
    number_of_samples: int = Field(
        default=100,
        description="Number of samples to generate"
    )
    context_column: str = Field(
        default="conversations",
        description="Column name containing context"
    )
    task_type: str = Field(
        default="qa",
        description="Type of task to generate"
    )
    sampling_params: Optional[SamplingParameters] = Field(
        default=None,
        description="Sampling parameters"
    )

class HuggingFaceUploadRequest(BaseModel):
    username: str = Field(description="Huggingface Hub username")
    dataset_name: str = Field(description="Name for the dataset on Huggingface Hub")

class BonitoResponse(BaseModel):
    column_names: List[str]
    num_samples: int
    message: str

# Initialize global Bonito instance
bonito_instance = None

@app.on_event("startup")
async def startup_event():
    """Initialize Bonito instance on startup"""
    global bonito_instance
    bonito_instance = Bonito("NousResearch/Genstruct-7B")

@app.post("/generate", response_model=BonitoResponse)
async def generate_synthetic_dataset(request: GenerationRequest):
    """Generate synthetic instruction tuning dataset"""
    global bonito_instance
    
    try:
        # Update model if different from current instance
        if bonito_instance is None or request.bonito_model != bonito_instance.model_name:
            bonito_instance = Bonito(request.bonito_model)
        
        # Load dataset
        unannotated_text = load_dataset(
            request.dataset,
            request.unannotated_text
        )[request.split].select(range(request.number_of_samples))
        
        # Set up sampling parameters
        sampling_params = request.sampling_params
        if sampling_params is None:
            sampling_params = SamplingParams(
                max_tokens=256,
                top_p=0.95,
                temperature=0.5,
                n=1
            )
        else:
            sampling_params = SamplingParams(
                max_tokens=sampling_params.max_tokens,
                top_p=sampling_params.top_p,
                temperature=sampling_params.temperature,
                n=sampling_params.n
            )
        
        # Generate synthetic dataset
        synthetic_dataset = bonito_instance.generate_tasks(
            unannotated_text,
            context_col=request.context_column,
            task_type=request.task_type,
            sampling_params=sampling_params
        )
        
        return BonitoResponse(
            column_names=synthetic_dataset.column_names,
            num_samples=len(synthetic_dataset),
            message="Dataset generated successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-to-hub")
async def upload_to_huggingface(
    request: HuggingFaceUploadRequest,
    synthetic_dataset=None
):
    """Upload generated dataset to Huggingface Hub"""
    try:
        if synthetic_dataset is None:
            raise HTTPException(
                status_code=400,
                detail="No dataset available. Generate one first using /generate endpoint"
            )
        
        # Push to hub
        synthetic_dataset.push_to_hub(f"{request.username}/{request.dataset_name}")
        
        return {
            "message": f"Dataset successfully uploaded to {request.username}/{request.dataset_name}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Configuration endpoint
@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "model": bonito_instance.model_name if bonito_instance else None,
        "available_task_types": ["qa", "classification", "generation"],  # Add more as needed
        "default_sampling_params": {
            "max_tokens": 256,
            "top_p": 0.95,
            "temperature": 0.5,
            "n": 1
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
