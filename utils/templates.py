relations = ['per:positive_impression', 'per:negative_impression', 'per:acquaintance', 'per:alumni', 'per:boss', 'per:subordinate', 'per:client', 'per:dates', 'per:friends', 'per:girl/boyfriend', 'per:neighbor', 'per:roommate', 'per:children', 'per:other_family', 'per:parents', 'per:siblings', 'per:spouse', 'per:place_of_residence',
             'per:place_of_birth', 'per:visited_place', 'per:origin', 'per:employee_or_member_of', 'per:schools_attended', 'per:works', 'per:age', 'per:date_of_birth', 'per:major', 'per:place_of_work', 'per:title', 'per:alternate_names', 'per:pet', 'gpe:residents_of_place', 'gpe:visitors_of_place', 'gpe:births_in_place', 'org:employees_or_members', 'org:students']


class QA_template():
    def __init__(self, include_inverse_relations=False):
        self.include_inverse_relations=include_inverse_relations
        self.relation_templates = {
            'per:positive_impression': [
                "Who or what does [ENTITY] have a positive impression of?",
                "[ENTITY] has a good opinion of what?",
                "Who does [ENTITY] feel positively about?"
                ],
            'per:negative_impression': [
                "Who does [ENTITY] have a negative impression of?",
                "[ENTITY] has a negative opinion of who?",
                "[ENTITY] really dislikes who?"
            ],
            'per:acquaintance': [
                "With whom is [ENTITY] acquainted?",
                "[ENTITY] is familiar with who?",
                "Who is an acquaintance of [ENTITY]?"
            ],
            'per:alumni': [
                "Who does [ENTITY] know from school?",
                "[ENTITY] was in school with who?",
                "What is the name of a person who went to school with [ENTITY]?"
            ],
            'per:boss': [
                "Who is the boss of [ENTITY]?",
                "What is the name of [ENTITY]'s supervisor?",
                "[ENTITY] is overseen by who at work?"
            ],
            'per:subordinate': [
                "[ENTITY] is the subordinate of who?",
                "Who does [ENTITY] work under?",
                "[ENTITY] works for who?"
            ],
            'per:client': [
                "Who is the client of [ENTITY]?",
                "[ENTITY] has a client named what?",
                "Who is a customer of [ENTITY]?"
            ],
            'per:dates': [
                "Who is [ENTITY] dating?",
                "[ENTITY] has gone out on dates with who?",
                "What is the name of the person that has dated [ENTITY]?"
            ],
            'per:friends': [
                "With whom does [ENTITY] have a friendship?",
                "Who is a friend of [ENTITY]?",
                "[ENTITY] is buddies with who?"
            ],
            'per:girl/boyfriend': [
                "Who is the girlfriend or boyfriend of [ENTITY]?",
                "Who is the fiancee of [ENTITY]?",
                "Who is [ENTITY]'s lover?"
            ],
            'per:neighbor': [
                "Who is the neighbor of [ENTITY]?",
                "[ENTITY] lives near who?",
                "Who is a nearby resident to [ENTITY]?"
            ],
            'per:roommate': [
                "Who does [ENTITY] have as a roommate?",
                "Who lives with [ENTITY]?",
                "[ENTITY] shares an apartment with who?"
            ],
            'per:children': [
                "Who is the child of [ENTITY]?",
                "What is the name of the son or daughter of [ENTITY]?",
                "[ENTITY] has offspring named what?"
            ],
            'per:other_family': [
                "Who is a distant family member of [ENTITY]?",
                "What is the name of the grandparent or grandchild of [ENTITY]?",
                "[ENTITY] has a cousin, aunt, uncle, or grandparent named what?"
            ],
            'per:parents': [
                "What is the name of [ENTITY]'s parent?",
                "[ENTITY] has a mother or father named what?",
                "Who is a parent of [ENTITY]?"
            ],
            'per:siblings': [
                "What is the name of [ENTITY]'s sibling?",
                "[ENTITY] has a brother or sister named what?",
                "Who is a brother or sister to [ENTITY]?"
            ],
            'per:spouse': [
                "Who is [ENTITY]'s spouse?",
                "[ENTITY] has a wife or husband named what?",
                "Who is married to [ENTITY]?"
            ],
            'per:place_of_residence': [
                "Where is [ENTITY] from?",
                "Where is the home of [ENTITY]?",
                "[ENTITY] is from what location?"
            ],
            'per:place_of_birth': [
                "Where was [ENTITY] born?",
                "What is the place of birth for [ENTITY]?",
                "[ENTITY] was born in what location?"
            ],
            'per:visited_place': [
                "What location has [ENTITY] visited?",
                "[ENTITY] traveled to what place?",
                "Where has [ENTITY] been to visit?"
            ],
            'per:origin': [
                "What is the origin of [ENTITY]?",
                "What is the nationality or ethnicity of [ENTITY]?",
                "[ENTITY] is descended from where?"
            ],
            'per:employee_or_member_of': [
                "[ENTITY] is an employee or member of what organization?",
                "What institution is [ENTITY] a member of?"
                "Which organization is [ENTITY] an employee or member of?"
            ],
            'per:schools_attended': [
                "What school did [ENTITY] attend?",
                "[ENTITY] went to what school?",
                "Which school did [ENTITY] take classes at?"
            ],
            'per:works': [
                "Where does [ENTITY] work?",
                "[ENTITY] works on what piece of culture?",
                "On what does [ENTITY] work?"
            ],
            'per:age': [
                "How old is [ENTITY]?",
                "What is [ENTITY]'s age?",
                "How long since [ENTITY] was born?"
            ],
            'per:date_of_birth': [
                "What is [ENTITY]'s date of birth?",
                "When was [ENTITY] born?",
                "When is [ENTITY]'s birthday?"
            ],
            'per:major': [
                "What topic did [ENTITY] study?",
                "[ENTITY] majored in which field?",
                "What is the field of study for [ENTITY]?"
            ],
            'per:place_of_work': [
                "Where does [ENTITY] work?",
                "[ENTITY] works at what location?",
                "What is the workplace of [ENTITY]?"
            ],
            'per:title': [
                "What is another title that [ENTITY] holds?",
                "[ENTITY] holds what employement or membership title?",
                "What is an alternate name for [ENTITY]'s position?"
            ],
            'per:alternate_names': [
                "What is an alternate name for [ENTITY]?",
                "[ENTITY] has what nickname or alias?",
                "How else can [ENTITY] be referred to?"
            ],
            'per:pet': [
                "What pet does [ENTITY] have?",
                "What is the name of [ENTITY]'s pet?",
                "[ENTITY] has what pet?"
            ],
            'gpe:residents_of_place': [
                "Who is a resident of [ENTITY]?",
                "Who calls or called [ENTITY] home?",
                "Who lives or lived in [ENTITY]?"
            ],
            'gpe:visitors_of_place': [
                "Who has visited [ENTITY]?",
                "[ENTITY] has been visited by whom?",
                "Who has traveled to [ENTITY]?"
            ],
            'gpe:births_in_place': [
                "Who was born in [ENTITY]?",
                "[ENTITY] is who's place of birth?",
                "Which person was born in [ENTITY]?"
            ],
            'org:employees_or_members': [
                "Who is an employee or member of [ENTITY]?",
                "Who belongs to [ENTITY]?",
                "[ENTITY] has which members?"
            ],
            'org:students': [
                "Who attended [ENTITY]?",
                "Who took classes at [ENTITY]?",
                "[ENTITY] had which person enrolled?"
            ]
        }
        self.inverse_relation_templates = {
            'per:positive_impression': [],
            'per:negative_impression': [],
            'per:acquaintance': [],
            'per:alumni': [],
            'per:boss': [],
            'per:subordinate': [],
            'per:client': [],
            'per:dates': [],
            'per:friends': [],
            'per:girl/boyfriend': [],
            'per:neighbor': [],
            'per:roommate': [],
            'per:children': [],
            'per:other_family': [],
            'per:parents': [],
            'per:siblings': [],
            'per:spouse': [],
            'per:place_of_residence': [],
            'per:place_of_birth': [],
            'per:visited_place': [],
            'per:origin': [],
            'per:employee_or_member_of': [],
            'per:schools_attended': [],
            'per:works': [],
            'per:age': [],
            'per:date_of_birth': [],
            'per:major': [],
            'per:place_of_work': [],
            'per:title': [],
            'per:alternate_names': [],
            'per:pet': [],
            'gpe:residents_of_place': [],
            'gpe:visitors_of_place': [],
            'gpe:births_in_place': [],
            'org:employees_or_members': [],
            'org:students': []
        }

    def fill_in_template(self, relation, entity1, entity2):
        filled_templates = []
        for template in self.relation_templates[relation]:
            filled_templates.append(template.replace("[ENTITY]", entity1))
        if self.include_inverse_relations:
            for template in self.inverse_relation_templates[relation]:
                filled_templates.append(template.replace("[ENTITY]", entity2))
        return filled_templates


class NLI_template():
    def __init__(self):
        self.relation_templates = {
            'per:positive_impression': [
                "[ENTITY1] has a positive impression of [ENTITY2].",
                "[ENTITY1] has a good opinion of [ENTITY2].",
                "[ENTITY1] feels positively about [ENTITY2]."
            ],
            'per:negative_impression': [
                "[ENTITY1] has a negative impression of [ENTITY2].",
                "[ENTITY1] has a negative opinion of [ENTITY2].",
                "[ENTITY1] really dislikes [ENTITY2]."
            ],
            'per:acquaintance': [
                "[ENTITY1] is acquainted with [ENTITY2].",
                "[ENTITY1] is familiar with [ENTITY2].",
                "[ENTITY2] is an acquaintance of [ENTITY1]."
            ],
            'per:alumni': [
                "[ENTITY1] knows [ENTITY2] from school.",
                "[ENTITY1] was in school with [ENTITY2].",
                "[ENTITY1] went to school with [ENTITY2]."
            ],
            'per:boss': [
                "[ENTITY2] is [ENTITY1]'s boss.",
                "[ENTITY2] is the supvervisor of [ENTITY1].",
                "[ENTITY2] oversees [ENTITY1] at work."
            ],
            'per:subordinate': [
                "[ENTITY1] is the subordinate of [ENTITY2].",
                "[ENTITY1] works under [ENTITY2].",
                "[ENTITY1] works for a person named [ENTITY2]."
            ],
            'per:client': [
                "[ENTITY2] is the client of [ENTITY1].",
                "[ENTITY1] has a client named [ENTITY2].",
                "[ENTITY2] is a customer of [ENTITY1]."
            ],
            'per:dates': [
                "[ENTITY1] and [ENTITY2] are dating.",
                "[ENTITY1] and [ENTITY2] have gone out on dates.",
                "[ENTITY2] has dated a person named [ENTITY1]."
            ],
            'per:friends': [
                "[ENTITY1] and [ENTITY2] are friends.",
                "[ENTITY2] has a friend named [ENTITY1].",
                "[ENTITY1] is buddies with [ENTITY2]."
            ],
            'per:girl/boyfriend': [
                "[ENTITY2] is the girlfriend or boyfriend of [ENTITY1].",
                "[ENTITY1] and [ENTITY2] are girlfriend and boyfriend.",
                "[ENTITY1] and [ENTITY2] are lovers."
            ],
            'per:neighbor': [
                "[ENTITY1] and [ENTITY2] are neighbors.",
                "[ENTITY1] and [ENTITY2] live close to each other.",
                "[ENTITY2] is a neighbor of [ENTITY1]."
            ],
            'per:roommate': [
                "[ENTITY1] is roommates with [ENTITY2].",
                "[ENTITY2] and [ENTITY1] live together.",
                "[ENTITY2] shares an apartment with [ENTITY1]."
            ],
            'per:children': [
                "[ENTITY2] is the child of [ENTITY1].",
                "[ENTITY1] has a son or daughter named [ENTITY2].",
                "[ENTITY1] has offspring named [ENTITY2]."
            ],
            'per:other_family': [
                "[ENTITY2] is a distant family member of [ENTITY1].",
                "[ENTITY1] has a grandparent or grandchild named [ENTITY2].",
                "[ENTITY2] is the cousin, aunt, uncle, or grandparent of [ENTITY1]."
            ],
            'per:parents': [
                "[ENTITY2] is [ENTITY1]'s parent.",
                "[ENTITY1] has a mother or father named [ENTITY2].",
                "[ENTITY2] is the parent of [ENTITY1]."
            ],
            'per:siblings': [
                "[ENTITY1] has a sibling named [ENTITY2].",
                "[ENTITY2] is the brother or sister of [ENTITY1].",
                "[ENTITY1] has a brother or sister named [ENTITY2]."
            ],
            'per:spouse': [
                "[ENTITY2] is [ENTITY1]'s spouse.",
                "[ENTITY1] has a husband or wife named [ENTITY2].",
                "[ENTITY1] is married to [ENTITY2]."
            ],
            'per:place_of_residence': [
                "[ENTITY1] is from [ENTITY2].",
                "[ENTITY1] lives in [ENTITY2].",
                "[ENTITY2] is home to [ENTITY1]."
            ],
            'per:place_of_birth': [
                "[ENTITY1] was born in [ENTITY2].",
                "[ENTITY2] is [ENTITY1]'s place of birth.",
                "[ENTITY1]'s birthplace is [ENTITY2]."
            ],
            'per:visited_place': [
                "[ENTITY1] visited [ENTITY2].",
                "[ENTITY1] traveled to [ENTITY2].",
                "[ENTITY1] has been to [ENTITY2] to visit."
            ],
            'per:origin': [
                "[ENTITY1]'s origin is [ENTITY2].",
                "The nationality or ethnicity of [ENTITY1] is [ENTITY2].",
                "[ENTITY1] is descended from [ENTITY2]."
            ],
            'per:employee_or_member_of': [
                "[ENTITY1] is an employee or member of [ENTITY2].",
                "[ENTITY1] is a member of the organization or instituion [ENTITY2].",
                "[ENTITY2] has the member [ENTITY1]."
            ],
            'per:schools_attended': [
                "[ENTITY1] attended [ENTITY2].",
                "[ENTITY1] went to the school [ENTITY2].",
                "[ENTITY1] took classes at [ENTITY2]."
            ],
            'per:works': [
                "[ENTITY1] works at [ENTITY2].",
                "[ENTITY1] works on the piece of culture [ENTITY2].",
                "[ENTITY2] is worked on by [ENTITY1]."
            ],
            'per:age': [
                "[ENTITY1] is [ENTITY2].",
                "[ENTITY1] has age [ENTITY2].",
                "It has [ENTITY2] since [ENTITY1] was born."
            ],
            'per:date_of_birth': [
                "The date of birth for [ENTITY1] is [ENTITY2].",
                "[ENTITY1] was born on [ENTITY2].",
                "[ENTITY2] is [ENTITY1]'s birthday."
            ],
            'per:major': [
                "[ENTITY1] studied [ENTITY2].",
                "[ENTITY1] majored in [ENTITY2].",
                "The field of study for [ENTITY1] was [ENTITY2]."
            ],
            'per:place_of_work': [
                "[ENTITY1] works at [ENTITY2].",
                "[ENTITY2] is the location where [ENTITY1] works.",
                "The workplace of [ENTITY1] is [ENTITY2]."
            ],
            'per:title': [
                "A title for [ENTITY1] is [ENTITY2].",
                "[ENTITY2] is an employement or membership title that [ENTITY1] holds.",
                "An alternate name for [ENTITY1]'s position is [ENTITY2]."
            ],
            'per:alternate_names': [
                "An altername name for [ENTITY1] is [ENTITY2].",
                "[ENTITY2] is an alias for [ENTITY1].",
                "[ENTITY1] can also be referred to as [ENTITY2]."
            ],
            'per:pet': [
                "[ENTITY2] is [ENTITY1]'s pet.",
                "[ENTITY1] has a pet named [ENTITY2].",
                "[ENTITY2] is a pet belonging to [ENTITY1]."
            ],
            'gpe:residents_of_place': [
                "[ENTITY2] is a resident of [ENTITY1].",
                "[ENTITY2] calls or called [ENTITY1] home.",
                "[ENTITY2] lives or lived in [ENTITY1]."
            ],
            'gpe:visitors_of_place': [
                "[ENTITY2] has visited [ENTITY1].",
                "The location [ENTITY1] was visited by [ENTITY2]."
                "[ENTITY2] has traveled to [ENTITY1]."
            ],
            'gpe:births_in_place': [
                "[ENTITY2] was born in [ENTITY1].",
                "[ENTITY1] is [ENTITY2]'s place of birth.",
                "[ENTITY2]'s birthplace is [ENTITY1]."
            ],
            'org:employees_or_members': [
                "[ENTITY1] has the employee or member [ENTITY2].",
                "[ENTITY2] is a member of the organization or instituion [ENTITY1].",
                "[ENTITY1] has the member [ENTITY2]."
            ],
            'org:students': [
                "[ENTITY2] attended [ENTITY1].",
                "[ENTITY2] took classes at [ENTITY1].",
                "[ENTITY1] had [ENTITY1] enrolled."
            ]
        }

    def fill_in_template(self, relation, entity1, entity2):
        filled_templates = []
        for template in self.relation_templates[relation]:
            filled_templates.append(template.replace(
                "[ENTITY1]", entity1).replace("[ENTITY2]", entity2))
        return filled_templates
