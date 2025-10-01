from src.envs.mario import MarioEnvConfig, MarioVectorEnvConfig, create_vector_env
import time

cfg = MarioVectorEnvConfig(
    num_envs=1,
    asynchronous=False,
    stage_schedule=((1, 1),),
    env=MarioEnvConfig(world=1, stage=1)
)

print("Creating environment...")
start_time = time.time()
try:
    env = create_vector_env(cfg)
    print(f"Environment created successfully in {time.time() - start_time:.2f}s")
    print("Resetting environment...")
    reset_start = time.time()
    obs, info = env.reset(seed=42)
    print(f"Environment reset successfully in {time.time() - reset_start:.2f}s")
    print("Observation shape:", obs.shape)
finally:
    env.close()
    print("Environment closed.")