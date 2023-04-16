import re
 
try:
    file = open("C:\\Users\\Admin\\Downloads\\emails_dirty.txt")
    for line in file:
        line = line.strip()
        emails = re.findall('\S+@\S+', line)
        if(len(emails) > 0):
            patn = re.sub(r"[\(['<,>)\]]", "", str(emails))
            print(patn)

except FileNotFoundError as e:
    print(e)
