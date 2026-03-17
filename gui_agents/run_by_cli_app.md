# 加了检索之前
python -m gui_agents.cli_app \
  --provider openai \
  --model gpt-5 \
  --model_url https://api.chatanywhere.tech/v1 \
  --model_api_key sk-ZtYubzCGDuXOL12HQtyZnN4Lp0cDyw25DFWk4eOK7AS88Nzw \
  --ground_provider openai \
  --ground_url https://api.chatanywhere.tech/v1 \
  --ground_api_key sk-ZtYubzCGDuXOL12HQtyZnN4Lp0cDyw25DFWk4eOK7AS88Nzw \
  --ground_model claude-sonnet-4-5-20250929 \
  --grounding_width 1280 \
  --grounding_height 720

  ---------------------------------------------------------------

python -m gui_agents.cli_app `
  --provider openai `
  --model gpt-5 `
  --model_url https://api.chatanywhere.tech/v1 `
  --model_api_key sk-ZtYubzCGDuXOL12HQtyZnN4Lp0cDyw25DFWk4eOK7AS88Nzw `
  --ground_provider openai `
  --ground_url https://api.chatanywhere.tech/v1 `
  --ground_api_key sk-ZtYubzCGDuXOL12HQtyZnN4Lp0cDyw25DFWk4eOK7AS88Nzw `
  --ground_model claude-sonnet-4-5-20250929 `
  --grounding_width 1280 `
  --grounding_height 720

python -m gui_agents.cli_app `
  --provider openai `
  --model qwen3-max `
  --model_url https://dashscope.aliyuncs.com/compatible-mode/v1 `
  --model_api_key sk-ff274f7282c344ef8d78a7c34a4d871d `
  --ground_provider openai `
  --ground_url https://api.chatanywhere.tech/v1 `
  --ground_api_key sk-ZtYubzCGDuXOL12HQtyZnN4Lp0cDyw25DFWk4eOK7AS88Nzw `
  --ground_model claude-sonnet-4-5-20250929 `
  --grounding_width 1280 `
  --grounding_height 720


# 加了检索之后
python -m gui_agents.cli_app `    
   --provider openai `
   --model gpt-5 `
   --model_url https://api.chatanywhere.tech/v1 `
   --model_api_key sk-ZtYubzCGDuXOL12HQtyZnN4Lp0cDyw25DFWk4eOK7AS88Nzw ` 
   --ground_provider openai `
   --ground_url https://api.chatanywhere.tech/v1 `
   --ground_api_key sk-ZtYubzCGDuXOL12HQtyZnN4Lp0cDyw25DFWk4eOK7AS88Nzw `
   --ground_model claude-sonnet-4-5-20250929 `
   --grounding_width 1280 `
   --grounding_height 720 `
   --enable_step_retrieval