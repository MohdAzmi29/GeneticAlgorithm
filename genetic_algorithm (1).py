
import random
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Genetic Algorithm Scheduling",
    page_icon="📺",
    layout="wide"
)

st.title("🧬 Genetic Algorithm Scheduling & Optimization System")

st.write(
    "Welcome! This application optimizes TV program schedules "
    "using Genetic Algorithms based on program ratings."
)


@st.cache_data
def load_data():
    return pd.read_csv("program_ratings.csv")


try:
    df = load_data()

except FileNotFoundError:
    st.error("program_ratings.csv not found.")
    st.stop()


program_column = df.columns[0]
rating_columns = df.columns[1:]

if len(rating_columns) == 0 or df.empty:
    st.error("Invalid dataset. Please check your CSV file.")
    st.stop()

df[rating_columns] = df[rating_columns].apply(
    pd.to_numeric, errors="coerce"
)

if df[rating_columns].isnull().any().any():
    st.error("Dataset contains missing or invalid ratings.")
    st.stop()

if df[program_column].isnull().any():
    st.error("Program names cannot be empty.")
    st.stop()

ratings = {}

for _, row in df.iterrows():
    program = str(row[program_column])
    ratings[program] = row[rating_columns].tolist()

all_programs = list(ratings.keys())

all_time_slots = list(
    range(6, 6 + len(rating_columns))
)

NUM_SLOTS = len(all_time_slots)

if len(all_programs) < 2:
    st.error("At least two programs are required.")
    st.stop()


st.sidebar.header("⚙️ Algorithm Configuration")

st.sidebar.write(
    "Tune your evolutionary parameters below:"
)
POP_SIZE = 50
GENERATIONS = 100
ELITISM_SIZE = 2

CROSS_RATE = st.sidebar.slider(
    "Crossover Rate (CO_R)",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.05
)

MUT_RATE = st.sidebar.slider(
    "Mutation Rate (MUT_RATE)",
    min_value=0.0,
    max_value=1.0,
    value=0.15,
    step=0.05
)



with st.expander("📊 View Raw Program Ratings Dataset (CSV)"):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )



def fitness_function(schedule):

    total_rating = 0

    for i, program in enumerate(schedule):

        total_rating += ratings[program][i]

    return total_rating


def initialize_population():

    population = []

    for _ in range(POP_SIZE):

        schedule = [
            random.choice(all_programs)
            for _ in range(NUM_SLOTS)
        ]

        population.append(schedule)

    return population



def selection(population):

    candidates = random.sample(
        population,
        min(3, len(population))
    )

    winner = max(
        candidates,
        key=fitness_function
    )

    return winner.copy()


def crossover(parent1, parent2):

    if NUM_SLOTS < 2:
        return parent1.copy(), parent2.copy()

    point = random.randint(1, NUM_SLOTS - 1)

    child1 = (
        parent1[:point] + parent2[point:]
    )

    child2 = (
        parent2[:point] + parent1[point:]
    )

    return child1, child2



def mutate(schedule):

    schedule = schedule.copy()

    for i in range(NUM_SLOTS):

        if random.random() < MUT_RATE:

            schedule[i] = random.choice(all_programs)

    return schedule


def genetic_algorithm(progress_bar, status_text):

    population = initialize_population()

    best_history = []

    best_schedule = None
    best_fitness = float("-inf")

    for generation in range(GENERATIONS):

        population.sort(
            key=fitness_function,
            reverse=True
        )

        current_best = population[0]
        current_fitness = fitness_function(current_best)

        if current_fitness > best_fitness:

            best_fitness = current_fitness
            best_schedule = current_best.copy()

        best_history.append(best_fitness)

        # Elitism
        new_population = [
            individual.copy()
            for individual in population[:ELITISM_SIZE]
        ]

        # Generate new population
        while len(new_population) < POP_SIZE:

            parent1 = selection(population)
            parent2 = selection(population)

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

        # Update progress
        progress = (generation + 1) / GENERATIONS

        progress_bar.progress(progress)

        status_text.write(
            f"Generation {generation + 1} / {GENERATIONS} "
            f"| Best Fitness: {best_fitness:.2f}"
        )

    # Check final population
    final_best = max(
        population,
        key=fitness_function
    )

    if fitness_function(final_best) > best_fitness:

        best_schedule = final_best.copy()
        best_fitness = fitness_function(final_best)

        best_history[-1] = best_fitness

    return best_schedule, best_history



if st.sidebar.button(
    "🚀 Run Optimization",
    use_container_width=True
):

    progress_bar = st.progress(0)

    status_text = st.empty()

    with st.spinner("Optimizing TV program schedule..."):

        best_schedule, best_history = genetic_algorithm(
            progress_bar,
            status_text
        )

    st.success("🎯 Optimization completed successfully!")


    st.header("🏆 Optimization Results")

    total_rating = fitness_function(best_schedule)

    average_rating = total_rating / NUM_SLOTS

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Fitness Score",
        f"{total_rating:.2f}"
    )

    col2.metric(
        "Programs Scheduled",
        NUM_SLOTS
    )

    col3.metric(
        "Average Rating",
        f"{average_rating:.3f}"
    )


    st.subheader("📺 Final Optimal TV Program Schedule")

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

else:

    st.info(
        "Adjust the algorithm parameters in the sidebar "
        "and click Run Optimization to generate the optimal schedule."
    )
