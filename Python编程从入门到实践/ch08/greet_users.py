def greet_users(names):
    """向列表的没问用户发出简单的问题"""
    for name in names:
        msg = f"Hello, {name.title()}!"
        print(msg)


usernames = ['hannah', 'ty', 'margot']
greet_users(usernames)
