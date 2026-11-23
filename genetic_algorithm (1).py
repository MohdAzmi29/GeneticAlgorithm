
import random
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Genetic Algorithm",
    page_icon="📺",
    layout="wide"
)

st.title("📺 Genetic Algorithm")
st.subheader("TV Program Scheduling Optimization")

@st.cache_data
def load_data():
    df = pd.read_csv("program_ratings.csv")
    return df

try:
    df = load_data()

    st.success("Dataset loaded successfully!")

    st.subheader("Program Ratings Dataset")
    st.dataframe(df, use_container_width=True)

except FileNotFoundError:
    st.error("program_ratings.csv not found in GitHub repository.")
    st.stop()

program_column = df.columns[0]
rating_columns = df.columns[1:]

if len(rating_columns) == 0:
    st.error("Your CSV must contain program ratings.")
    st.stop()

df[rating_columns] = df[rating_columns].apply(
    pd.to_numeric, errors="coerce"
)

if df[rating_columns].isnull().any().any():
    st.error("Your dataset contains missing or invalid ratings.")
    st.stop()

ratings = {}

for _, row in df.iterrows():
    program = row[program_column]
    ratings[program] = row[rating_columns].tolist()

all_programs = list(ratings.keys())
all_time_slots = list(range(6, 6 + len(rating_columns)))

NUM_SLOTS = len(all_time_slots)

if len(all_programs) < NUM_SLOTS:
    st.error(
        f"You need at least {NUM_SLOTS} programs "
        f"to fill all time slots without repeating programs."
    )
    st.stop()

st.sidebar.header("Algorithm Parameters")

POP_SIZE = st.sidebar.slider(
    "Population Size", 10, 200, 50
)

GENERATIONS = st.sidebar.slider(
    "Number of Generations", 10, 500, 100
)

MUT_RATE = st.sidebar.slider(
    "Mutation Rate", 0.0, 1.0, 0.2, 0.05
)

CROSS_RATE = st.sidebar.slider(
    "Crossover Rate", 0.0, 1.0, 0.8, 0.05
)

ELITISM_SIZE = st.sidebar.slider(
    "Elitism Size", 1, 10, 2
)

if ELITISM_SIZE >= POP_SIZE:
    st.error("Elitism size must be smaller than population size.")
    st.stop()

def fitness_function(schedule):

    total_rating = 0

    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]

    return total_rating


def initialize_population():

    population = []

    for _ in range(POP_SIZE):

        schedule = random.sample(
            all_programs, NUM_SLOTS
        )

        population.append(schedule)

    return population



def selection(population):

    selected = []

    for _ in range(POP_SIZE):

        candidates = random.sample(
            population, min(3, len(population))
        )

        winner = max(
            candidates,
            key=fitness_function
        )

        selected.append(winner)

    return selected


def crossover(parent1, parent2):

    if NUM_SLOTS < 2:
        return parent1.copy(), parent2.copy()

    point1, point2 = sorted(
        random.sample(range(NUM_SLOTS), 2)
    )

    def create_child(p1, p2):

        child = [None] * NUM_SLOTS

        child[point1:point2 + 1] = (
            p1[point1:point2 + 1]
        )

        used = set(child[point1:point2 + 1])

        candidates = (
            p2[point2 + 1:] +
            p2[:point2 + 1] +
            all_programs
        )

        empty_positions = (
            list(range(point2 + 1, NUM_SLOTS)) +
            list(range(0, point1))
        )

        for position in empty_positions:

            for program in candidates:

                if program not in used:

                    child[position] = program
                    used.add(program)
                    break

        return child

    child1 = create_child(parent1, parent2)
    child2 = create_child(parent2, parent1)

    return child1, child2


def mutate(schedule):

    schedule = schedule.copy()

    if NUM_SLOTS >= 2:

        if random.random() < MUT_RATE:

            pos1, pos2 = random.sample(
                range(NUM_SLOTS), 2
            )

            schedule[pos1], schedule[pos2] = (
                schedule[pos2],
                schedule[pos1]
            )

    return schedule



def genetic_algorithm():

    population = initialize_population()

    best_history = []

    for generation in range(GENERATIONS):

        population.sort(
            key=fitness_function,
            reverse=True
        )

        new_population = population[:ELITISM_SIZE]

        selected = selection(population)

        while len(new_population) < POP_SIZE:

            parent1, parent2 = random.sample(
                selected, 2
            )

            if random.random() < CROSS_RATE:

                child1, child2 = crossover(
                    parent1, parent2
                )

            else:

                child1 = parent1.copy()
                child2 = parent2.copy()

            child1 = mutate(child1)
            child2 = mutate(child2)

            new_population.append(child1)

            if len(new_population) < POP_SIZE:
                new_population.append(child2)

        population = new_population

        best = max(
            population,
            key=fitness_function
        )

        best_history.append(fitness_function(best))

    best_schedule = max(
        population,
        key=fitness_function
    )

    return best_schedule, best_history



if st.button("🚀 Generate Optimal Schedule"):

    with st.spinner("Optimizing TV program schedule..."):

        best_schedule, best_history = genetic_algorithm()

    st.success("Optimization completed successfully!")

    total_rating = fitness_function(best_schedule)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Rating",
        f"{total_rating:.2f}"
    )

    col2.metric(
        "Programs Scheduled",
        len(best_schedule)
    )

    col3.metric(
        "Generations",
        GENERATIONS
    )

    st.subheader("🏆 Final Optimal Schedule")

    results = []

    for i, program in enumerate(best_schedule):

        results.append({
            "Time Slot": f"{all_time_slots[i]:02d}:00",
            "Program": program,
            "Rating": ratings[program][i]
        })

    results_df = pd.DataFrame(results)

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )


    st.subheader("📈 Genetic Algorithm Progress")

    progress_df = pd.DataFrame({
        "Generation": range(1, GENERATIONS + 1),
        "Best Fitness": best_history
    })

    st.line_chart(
        progress_df,
        x="Generation",
        y="Best Fitness"
    )

    csv_output = results_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Optimal Schedule",
        data=csv_output,
        file_name="optimal_schedule.csv",
        mime="text/csv"
    )
