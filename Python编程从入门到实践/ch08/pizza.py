def make_pizza(size, *toppings):
    print(f"\nMaking a {size}-inchpizza with the following toppings:")
    for topping in toppings:
        print(f"- {topping}")
