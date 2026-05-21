$env:MODEL_API_KEY="sk-e7662b60dd0816fc91306166d1d7930390d52989b850bbb610e945068f005f12"
$env:MODEL_URL="http://localhost:8080/v1"
$env:GROUND_API_KEY="sk-ZtYubzCGDuXOL12HQtyZnN4Lp0cDyw25DFWk4eOK7AS88Nzw"
$env:GROUND_URL="https://api.chatanywhere.tech/v1"
$env:SILICONFLOW_API_KEY="sk-ytwgqxhcywsszqqhuzuulsmcyslmoycivfdzlsctevuqruzc"
$env:QWEN_API_KEY="sk-4d920336c17e438f8c70e10c02f2ad83"

D:\develop\miniconda3\envs\osworld\python.exe -m gui_agents.cli_app `
  --provider openai `
  --model gpt-5.2 `
  --model_url $env:MODEL_URL `
  --model_api_key $env:MODEL_API_KEY `
  --ground_provider openai `
  --ground_url GROUND_URL `
  --ground_api_key GROUND_API_KEY `
  --ground_model claude-sonnet-4-5-20250514 `
  --grounding_width 1280 `
  --grounding_height 720 `
  --enable_cu_retrieval