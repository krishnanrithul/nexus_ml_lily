from query_engine import ask
from datetime import datetime

QUESTIONS = [
    # Original 5
    "How reliable are the model's predictions and where does it struggle?",
    "Which earnings periods had the largest prediction errors?",
    "What factors are most responsible for driving prediction accuracy?",
    "Is earnings sentiment influencing the model's predictions?",
    "Which predictions had the largest errors and what caused them?",

    # Director-level 5
    "Which earnings quarters should we have the least confidence in?",
    "Were macro factors like VIX and unemployment driving our misses or was it analyst signals?",
    "Is the model more accurate in recent quarters than earlier ones?",
    "How much is earnings sentiment contributing to the model's view?",
    "Which specific earnings events caught the model most off guard and why?",
]

output_file = "output/demo_results.txt"

with open(output_file, "w") as f:
    header = f"NexusML Demo — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n" + "="*60 + "\n\n"
    f.write(header)
    print(header)

    for i, question in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] Asking: {question}")

        answer, intent, results = ask(question)

        block = (
            f"Q{i}: {question}\n"
            f"Intent routed: {intent}\n"
            f"Sources retrieved: {len(results)}\n\n"
            f"Answer:\n{answer}\n\n"
            f"{'-'*60}\n\n"
        )

        f.write(block)
        print(block)

print(f"✅ Results saved to {output_file}")