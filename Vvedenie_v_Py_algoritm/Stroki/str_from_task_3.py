total = 500 #int(input())
salary = 100 #int(input())
part = salary / total
template = "{} составляет {:.0%}% от {}".format(salary, part, total)
print(template)