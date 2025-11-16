# CONFIG FIX REQUIRED

## Error
```
ValueError: Invalid model type: gemini-1.5-flash. Choose 'openai' or 'gemini'
```

## Problem
Your local `config.py` file has incorrect value for `AI_MODEL`.

## Fix

Open your local file:
```
C:\Users\TROLL\Desktop\Upwork\artem\10postcheck\config.py
```

Find this line:
```python
AI_MODEL = 'gemini-1.5-flash'  # ❌ WRONG
```

Change it to:
```python
AI_MODEL = 'gemini'  # ✅ CORRECT
```

## Explanation

`AI_MODEL` only accepts two values:
- `'openai'` → Use OpenAI (GPT models)
- `'gemini'` → Use Google Gemini

The actual model name is specified separately:
- For OpenAI: `OPENAI_MODEL = 'gpt-5-nano'`
- For Gemini: `GEMINI_MODEL = 'gemini-2.0-flash-exp'`

## Your Correct Config Should Be:

```python
AI_MODEL = 'gemini'              # ← Which AI provider to use
GEMINI_MODEL = 'gemini-2.0-flash-exp'  # ← Which specific model
```

After fixing, run:
```bash
python test_quick.py
```
