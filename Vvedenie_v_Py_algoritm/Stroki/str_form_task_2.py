template = """Уважаемый {},
Банк {} приглашает вас в отделение банка №{} для получения вашей карты.
{}"""
customer = 'Иванов И.' #input()
bank = 'Hudoy' #input()
department_num = '001' #input()
dt = '01.01.2023' #input()
print(template.format(customer, bank, department_num, dt))