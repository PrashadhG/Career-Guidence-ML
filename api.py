from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import random
from typing import List, Dict, Optional, Union, Any
import google.generativeai as genai
import base64
from datetime import datetime
from collections import Counter
from fastapi.middleware.cors import CORSMiddleware 
from dotenv import load_dotenv
import os

app = FastAPI()

load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change to specific origins as needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Load ML models and data
kmeans = joblib.load('kmeans_career.pkl')
with open('career_clusters.json', 'r') as f:
    career_clusters = json.load(f)

# Load psychometric question banks
with open('orientations.txt', 'r') as f:
    orientation_questions = json.load(f)

with open('interest.txt', 'r') as f:
    interest_questions = json.load(f)

with open('personality.txt', 'r') as f:
    personality_questions = json.load(f)

# Career recommendations dictionary
career_recommendations = {
    'PCM': {
        'subjects': {
            'core': ['Physics', 'Chemistry', 'Mathematics'],
            'electives': ['Computer Science', 'Engineering Graphics', 'Electronics']
        },
        'skills': {
            'technical': [
                'Programming Languages (Python, Java)',
                'Mathematics & Statistics',
                'Problem Solving',
                'CAD Software',
                'Data Analysis'
            ],
            'soft': [
                'Analytical Thinking',
                'Project Management',
                'Team Collaboration',
                'Technical Documentation',
                'Time Management'
            ]
        },
        'activity_areas': [
            'coding',
            'computer_science',
            'data_analysis',
            'engineering_design',
            'mathematics'
        ]
    },
    'PCB': {
        'subjects': {
            'core': ['Physics', 'Chemistry', 'Biology'],
            'electives': ['Biotechnology', 'Psychology', 'Health Science']
        },
        'skills': {
            'technical': [
                'Laboratory Techniques',
                'Research Methodology',
                'Medical Terminology',
                'Data Analysis',
                'Scientific Documentation'
            ],
            'soft': [
                'Critical Thinking',
                'Attention to Detail',
                'Communication',
                'Ethics & Compliance',
                'Observation Skills'
            ]
        },
        'activity_areas': [
            'biology_experiments',
            'healthcare',
            'research_analysis',
            'scientific_method',
            'laboratory_procedures'
        ]
    },
    'Commerce': {
        'subjects': {
            'core': ['Accountancy', 'Business Studies', 'Economics'],
            'electives': ['Mathematics', 'Financial Markets', 'Marketing']
        },
        'skills': {
            'technical': [
                'Financial Analysis',
                'Business Software',
                'Market Research',
                'Data Analytics',
                'Digital Marketing'
            ],
            'soft': [
                'Business Communication',
                'Negotiation',
                'Leadership',
                'Problem Solving',
                'Decision Making'
            ]
        },
        'activity_areas': [
            'business_case_studies',
            'financial_analysis',
            'market_research',
            'entrepreneurship',
            'management'
        ]
    },
    'Humanities': {
        'subjects': {
            'core': ['History', 'Political Science', 'Sociology'],
            'electives': ['Psychology', 'Economics', 'Literature']
        },
        'skills': {
            'technical': [
                'Research Methods',
                'Content Writing',
                'Social Media Management',
                'Data Analysis',
                'Project Planning'
            ],
            'soft': [
                'Critical Thinking',
                'Communication',
                'Empathy',
                'Cultural Awareness',
                'Analysis & Interpretation'
            ]
        },
        'activity_areas': [
            'essay_writing',
            'critical_analysis',
            'social_research',
            'creative_projects',
            'debate_argumentation'
        ]
    }
}

# Career-specific activity mapping
career_activity_mapping = {
    # PCM career activities
    'Software Engineer': ['coding', 'computer_science', 'problem_solving'],
    'Data Scientist': ['data_analysis', 'mathematics', 'coding'],
    'Architect': ['engineering_design', 'spatial_planning', 'mathematics'],
    'Mechanical Engineer': ['engineering_design', 'physics_applications', 'mathematics'],
    'Electrical Engineer': ['circuit_design', 'electronics', 'mathematics'],
    'Civil Engineer': ['structural_design', 'mathematics', 'spatial_planning'],
    
    # PCB career activities
    'Doctor': ['healthcare', 'biology_experiments', 'research_analysis'],
    'Biologist': ['biology_experiments', 'scientific_method', 'research_analysis'],
    'Pharmacist': ['healthcare', 'laboratory_procedures', 'chemistry_applications'],
    'Veterinarian': ['biology_experiments', 'healthcare', 'animal_sciences'],
    'Nurse': ['healthcare', 'patient_care', 'medical_knowledge'],
    'Biotechnologist': ['biology_experiments', 'laboratory_procedures', 'research_analysis'],
    
    # Commerce career activities
    'Accountant': ['financial_analysis', 'business_case_studies', 'mathematics'],
    'Marketing Manager': ['market_research', 'business_case_studies', 'creative_projects'],
    'Financial Analyst': ['financial_analysis', 'data_analysis', 'mathematics'],
    'Entrepreneur': ['entrepreneurship', 'management', 'business_case_studies'],
    'Business Consultant': ['business_case_studies', 'market_research', 'management'],
    'Investment Banker': ['financial_analysis', 'market_research', 'mathematics'],
    
    # Humanities career activities
    'Lawyer': ['debate_argumentation', 'critical_analysis', 'essay_writing'],
    'Journalist': ['essay_writing', 'critical_analysis', 'research_analysis'],
    'Psychologist': ['social_research', 'critical_analysis', 'healthcare'],
    'Historian': ['essay_writing', 'critical_analysis', 'research_analysis'],
    'Teacher': ['curriculum_design', 'creative_projects', 'education_planning'],
    'Social Worker': ['social_research', 'healthcare', 'case_management'],
    
    # Creative career activities
    'Graphic Designer': ['visual_design', 'creative_projects', 'digital_illustration'],
    'UX Designer': ['user_research', 'visual_design', 'prototyping'],
    'Content Writer': ['essay_writing', 'creative_writing', 'content_strategy'],
    'Photographer': ['visual_composition', 'creative_projects', 'digital_editing'],
    'Video Editor': ['digital_editing', 'storytelling', 'creative_projects'],
    'Fashion Designer': ['visual_design', 'creative_projects', 'material_knowledge']
}

# Career trait mappings based on assessment categories
career_trait_mappings = {
    # Orientation traits
    'orientation': {
        'structure_preference': {
            'structured': ['Accountant', 'Doctor', 'Engineer', 'Lawyer', 'Financial Analyst', 'Pharmacist'],
            'flexible': ['Entrepreneur', 'Artist', 'Designer', 'Writer', 'Photographer', 'UX Designer'],
            'mixed': ['Teacher', 'Manager', 'Consultant', 'Researcher', 'Marketing Manager', 'Journalist']
        },
        'value_priority': {
            'security': ['Accountant', 'Civil Servant', 'Engineer', 'Doctor', 'Financial Analyst', 'Nurse'],
            'creativity': ['Designer', 'Writer', 'Artist', 'Architect', 'Photographer', 'Content Writer'],
            'helping': ['Teacher', 'Social Worker', 'Nurse', 'Counselor', 'Doctor', 'Veterinarian'],
            'challenge': ['Entrepreneur', 'Researcher', 'Consultant', 'Software Engineer', 'Data Scientist', 'Investment Banker']
        },
        'work_pace': {
            'steady': ['Accountant', 'Researcher', 'Librarian', 'Analyst', 'Editor', 'Pharmacist'],
            'fast': ['Journalist', 'Entrepreneur', 'Emergency Services', 'Event Planner', 'Marketing Manager', 'Stockbroker'],
            'varying': ['Teacher', 'Manager', 'Consultant', 'Software Engineer', 'Lawyer', 'Psychologist']
        }
    },
    # Interest traits
    'interest': {
        'field_preference': {
            'technical': ['Engineer', 'Software Developer', 'Data Scientist', 'Architect', 'Systems Analyst', 'Network Administrator'],
            'creative': ['Designer', 'Writer', 'Artist', 'Photographer', 'Filmmaker', 'Musician'],
            'helping': ['Doctor', 'Teacher', 'Social Worker', 'Counselor', 'Nurse', 'Therapist'],
            'business': ['Manager', 'Entrepreneur', 'Consultant', 'Financial Analyst', 'Marketing Director', 'Investment Banker']
        },
        'domain_focus': {
            'humanities': ['Writer', 'Historian', 'Teacher', 'Journalist', 'Philosopher', 'Linguist'],
            'stem': ['Engineer', 'Scientist', 'Data Analyst', 'Researcher', 'Mathematician', 'Physician'],
            'commerce': ['Accountant', 'Marketing Manager', 'Financial Analyst', 'Economist', 'Business Consultant', 'Entrepreneur'],
            'arts': ['Designer', 'Artist', 'Musician', 'Architect', 'Photographer', 'Filmmaker']
        },
        'activity_preference': {
            'analytical': ['Data Analyst', 'Researcher', 'Scientist', 'Engineer', 'Financial Analyst', 'Actuary'],
            'hands_on': ['Mechanic', 'Surgeon', 'Chef', 'Craftsperson', 'Physical Therapist', 'Dental Hygienist'],
            'creative': ['Designer', 'Writer', 'Artist', 'Architect', 'Content Creator', 'Advertising Executive'],
            'social': ['Teacher', 'Counselor', 'Sales Representative', 'Manager', 'Human Resources', 'Event Planner']
        }
    },
    # Personality traits
    'personality': {
        'work_style': {
            'independent': ['Researcher', 'Writer', 'Artist', 'Analyst', 'Programmer', 'Accountant'],
            'collaborative': ['Manager', 'Teacher', 'Marketing Professional', 'Event Planner', 'Human Resources', 'Sales Representative'],
            'mixed': ['Consultant', 'Engineer', 'Designer', 'Doctor', 'Lawyer', 'Entrepreneur']
        },
        'problem_approach': {
            'analytical': ['Scientist', 'Engineer', 'Analyst', 'Doctor', 'Researcher', 'Lawyer'],
            'creative': ['Designer', 'Marketer', 'Writer', 'Entrepreneur', 'Architect', 'Artist'],
            'pragmatic': ['Manager', 'Technician', 'Accountant', 'Lawyer', 'Financial Planner', 'Administrator']
        },
        'social_orientation': {
            'people_focused': ['Teacher', 'HR Manager', 'Social Worker', 'Sales Representative', 'Counselor', 'Customer Service'],
            'task_focused': ['Programmer', 'Accountant', 'Analyst', 'Engineer', 'Researcher', 'Editor']
        },
        'leadership_style': {
            'directive': ['Executive', 'Military Officer', 'Athletic Coach', 'Project Manager', 'Surgeon', 'Judge'],
            'supportive': ['Team Leader', 'School Principal', 'Mentor', 'Product Manager', 'Nurse Manager', 'Social Work Director'],
            'delegative': ['Research Director', 'Creative Director', 'Department Head', 'CEO', 'University Dean', 'Film Director']
        }
    },
    # Aptitude traits based on cognitive test scores
    'aptitude': {
        'numerical_aptitude': {
            'high': ['Accountant', 'Engineer', 'Data Scientist', 'Financial Analyst', 'Actuary', 'Mathematician'],
            'medium': ['Teacher', 'Marketing Analyst', 'Project Manager', 'Business Analyst', 'Pharmacist', 'Healthcare Administrator'],
            'low': ['Writer', 'Designer', 'Social Worker', 'Artist', 'Counselor', 'Child Care Worker']
        },
        'spatial_aptitude': {
            'high': ['Architect', 'Surgeon', 'Graphic Designer', 'Civil Engineer', 'Aircraft Pilot', 'Industrial Designer'],
            'medium': ['Photographer', 'Interior Designer', 'Mechanic', 'Nurse', 'Chef', 'Construction Manager'],
            'low': ['Accountant', 'Writer', 'Customer Service', 'Sales Representative', 'Call Center Operator', 'Data Entry Specialist']
        },
        'perceptual_aptitude': {
            'high': ['Detective', 'Quality Control', 'Editor', 'Software Tester', 'Air Traffic Controller', 'Medical Diagnostician'],
            'medium': ['Researcher', 'Teacher', 'Administrator', 'Technician', 'Journalist', 'Laboratory Technician'],
            'low': ['Public Speaker', 'Sales Representative', 'Manager', 'Host', 'Event Coordinator', 'Tour Guide']
        },
        'abstract_reasoning': {
            'high': ['Scientist', 'Philosopher', 'Software Developer', 'Strategist', 'Research Analyst', 'AI Specialist'],
            'medium': ['Teacher', 'Manager', 'Analyst', 'Marketing Professional', 'Counselor', 'Human Resources'],
            'low': ['Construction Worker', 'Cashier', 'Receptionist', 'Delivery Driver', 'Security Guard', 'Factory Worker']
        },
        'verbal_reasoning': {
            'high': ['Lawyer', 'Writer', 'Professor', 'Journalist', 'Editor', 'Public Relations Specialist'],
            'medium': ['Teacher', 'Manager', 'Customer Service', 'Sales Representative', 'Consultant', 'Administrator'],
            'low': ['Engineer', 'Mathematician', 'Mechanic', 'Technician', 'Tradesperson', 'Machine Operator']
        }
    }
}

skill_gap_assessment = {
    # Technical skills
    'coding': {
        'excellent': ['Advanced programming concepts', 'Multiple language proficiency', 'Complex problem solving'],
        'good': ['Basic programming concepts', 'Single language proficiency', 'Simple problem solving'],
        'needs_improvement': ['Syntax fundamentals', 'Algorithm basics', 'Structured thinking']
    },
    'computer_science': {
        'excellent': ['Advanced data structures', 'System architecture', 'Algorithm optimization'],
        'good': ['Basic data structures', 'Computer organization', 'Algorithm analysis'],
        'needs_improvement': ['CS fundamentals', 'Basic algorithms', 'Computing concepts']
    },
    'data_analysis': {
        'excellent': ['Statistical modeling', 'Advanced visualization', 'Machine learning'],
        'good': ['Basic statistics', 'Data visualization', 'SQL querying'],
        'needs_improvement': ['Data literacy', 'Basic Excel', 'Chart interpretation']
    },
    'engineering_design': {
        'excellent': ['Advanced CAD', '3D modeling', 'Design optimization'],
        'good': ['Basic CAD', 'Blueprint interpretation', 'Design principles'],
        'needs_improvement': ['Sketching', 'Measurement', 'Spatial reasoning']
    },
    'mathematics': {
        'excellent': ['Calculus', 'Linear algebra', 'Statistical analysis'],
        'good': ['Algebra', 'Geometry', 'Probability'],
        'needs_improvement': ['Basic calculations', 'Mathematical reasoning', 'Number sense']
    },
    
    # Science and healthcare skills
    'biology_experiments': {
        'excellent': ['Complex experiment design', 'Lab techniques', 'Results analysis'],
        'good': ['Basic experiment design', 'Lab safety', 'Observation skills'],
        'needs_improvement': ['Scientific method', 'Basic lab procedures', 'Data recording']
    },
    'healthcare': {
        'excellent': ['Medical terminology', 'Patient care', 'Diagnostic reasoning'],
        'good': ['Basic anatomy', 'Health protocols', 'Care principles'],
        'needs_improvement': ['Body systems', 'Health basics', 'First aid concepts']
    },
    'research_analysis': {
        'excellent': ['Research methodology', 'Data interpretation', 'Critical analysis'],
        'good': ['Information literacy', 'Source evaluation', 'Basic analysis'],
        'needs_improvement': ['Research questions', 'Information gathering', 'Summary skills']
    },
    'scientific_method': {
        'excellent': ['Hypothesis testing', 'Experimental control', 'Variable manipulation'],
        'good': ['Basic scientific process', 'Controlled experiments', 'Data collection'],
        'needs_improvement': ['Scientific inquiry', 'Observation', 'Question formulation']
    },
    'laboratory_procedures': {
        'excellent': ['Advanced lab techniques', 'Equipment calibration', 'Protocol development'],
        'good': ['Basic lab techniques', 'Equipment operation', 'Protocol following'],
        'needs_improvement': ['Lab safety', 'Tool identification', 'Procedure steps']
    },
    
    # Business and commerce skills
    'business_case_studies': {
        'excellent': ['Industry analysis', 'Strategic recommendations', 'Multi-variable problem solving'],
        'good': ['Business model understanding', 'Problem identification', 'Solution development'],
        'needs_improvement': ['Business terminology', 'Case reading', 'Basic analysis']
    },
    'financial_analysis': {
        'excellent': ['Financial modeling', 'Investment analysis', 'Risk assessment'],
        'good': ['Financial statement analysis', 'Ratio analysis', 'Budgeting'],
        'needs_improvement': ['Basic accounting', 'Financial literacy', 'Calculation skills']
    },
    'market_research': {
        'excellent': ['Advanced survey design', 'Market segmentation', 'Competitive analysis'],
        'good': ['Basic survey techniques', 'Market trends', 'Consumer behavior'],
        'needs_improvement': ['Research basics', 'Data collection', 'Trend identification']
    },
    'entrepreneurship': {
        'excellent': ['Business model innovation', 'Strategic planning', 'Resource allocation'],
        'good': ['Business plan development', 'Market opportunity', 'Value proposition'],
        'needs_improvement': ['Business basics', 'Idea generation', 'Market awareness']
    },
    'management': {
        'excellent': ['Strategic leadership', 'Team development', 'Organizational change'],
        'good': ['Team coordination', 'Resource management', 'Performance evaluation'],
        'needs_improvement': ['Management principles', 'Communication', 'Planning skills']
    },
    
    # Humanities and writing skills
    'essay_writing': {
        'excellent': ['Argumentative writing', 'Research integration', 'Stylistic elements'],
        'good': ['Thesis development', 'Supporting evidence', 'Logical flow'],
        'needs_improvement': ['Basic structure', 'Grammar and syntax', 'Idea organization']
    },
    'critical_analysis': {
        'excellent': ['Theoretical frameworks', 'Interdisciplinary connections', 'Original insights'],
        'good': ['Source evaluation', 'Argument assessment', 'Evidence analysis'],
        'needs_improvement': ['Main idea identification', 'Supporting evidence', 'Inference skills']
    },
    'social_research': {
        'excellent': ['Research design', 'Qualitative & quantitative methods', 'Analysis frameworks'],
        'good': ['Interview techniques', 'Survey design', 'Data coding'],
        'needs_improvement': ['Research ethics', 'Basic methods', 'Data collection']
    },
    'creative_projects': {
        'excellent': ['Innovative concepts', 'Execution mastery', 'Cross-medium integration'],
        'good': ['Project planning', 'Creative execution', 'Self-evaluation'],
        'needs_improvement': ['Idea generation', 'Basic techniques', 'Project completion']
    },
    'debate_argumentation': {
        'excellent': ['Advanced argumentation', 'Counterargument anticipation', 'Persuasive rhetoric'],
        'good': ['Logical reasoning', 'Evidence use', 'Rebuttal skills'],
        'needs_improvement': ['Argument structure', 'Active listening', 'Position articulation']
    },
    
    # Creative and design skills
    'visual_design': {
        'excellent': ['Advanced composition', 'Design systems', 'Visual storytelling'],
        'good': ['Basic design principles', 'Color theory', 'Layout techniques'],
        'needs_improvement': ['Visual elements', 'Basic tools', 'Design fundamentals']
    },
    'digital_illustration': {
        'excellent': ['Advanced digital techniques', 'Stylistic development', 'Complex compositions'],
        'good': ['Basic illustration tools', 'Digital coloring', 'Simple compositions'],
        'needs_improvement': ['Tool basics', 'Drawing fundamentals', 'Digital workflow']
    },
    'user_research': {
        'excellent': ['User testing methodology', 'Behavioral analysis', 'Research synthesis'],
        'good': ['Interview techniques', 'Usability principles', 'User personas'],
        'needs_improvement': ['Research basics', 'User needs', 'Feedback collection']
    },
    'prototyping': {
        'excellent': ['High-fidelity prototyping', 'Interactive workflows', 'User testing'],
        'good': ['Mid-fidelity mockups', 'Basic interactions', 'Wireframing'],
        'needs_improvement': ['Sketching interfaces', 'Basic layouts', 'Tool familiarity']
    },
    'storytelling': {
        'excellent': ['Narrative structure', 'Character development', 'Thematic elements'],
        'good': ['Story arcs', 'Basic character building', 'Plot development'],
        'needs_improvement': ['Story basics', 'Descriptive writing', 'Narrative flow']
    },
    'creative_writing': {
        'excellent': ['Voice development', 'Genre mastery', 'Advanced techniques'],
        'good': ['Creative elements', 'Genre awareness', 'Effective language'],
        'needs_improvement': ['Writing basics', 'Idea development', 'Language usage']
    },
    'visual_composition': {
        'excellent': ['Advanced composition', 'Visual storytelling', 'Technical mastery'],
        'good': ['Basic composition rules', 'Visual elements', 'Technical understanding'],
        'needs_improvement': ['Composition basics', 'Technical foundations', 'Visual awareness']
    },
    'digital_editing': {
        'excellent': ['Advanced editing techniques', 'Workflow optimization', 'Style development'],
        'good': ['Basic editing tools', 'Correction techniques', 'Simple effects'],
        'needs_improvement': ['Software basics', 'Basic adjustments', 'Tool familiarity']
    },
    
    # Other specialized skills
    'spatial_planning': {
        'excellent': ['Complex spatial relationships', 'Site analysis', 'Environmental integration'],
        'good': ['Basic spatial planning', 'Site considerations', 'Functional layouts'],
        'needs_improvement': ['Spatial awareness', 'Basic mapping', 'Scale understanding']
    },
    'circuit_design': {
        'excellent': ['Complex circuit systems', 'Circuit optimization', 'Electronic theory'],
        'good': ['Basic circuit design', 'Component integration', 'Circuit analysis'],
        'needs_improvement': ['Circuit fundamentals', 'Component identification', 'Basic electricity']
    },
    'electronics': {
        'excellent': ['Advanced electronic systems', 'Troubleshooting', 'System integration'],
        'good': ['Basic electronic concepts', 'Component usage', 'Simple circuits'],
        'needs_improvement': ['Electronic fundamentals', 'Component recognition', 'Safety procedures']
    },
    'structural_design': {
        'excellent': ['Complex structural systems', 'Structural analysis', 'Material optimization'],
        'good': ['Basic structural principles', 'Load considerations', 'Material selection'],
        'needs_improvement': ['Structural basics', 'Material properties', 'Simple supports']
    },
    'physics_applications': {
        'excellent': ['Applied physics modeling', 'Experimental design', 'Data analysis'],
        'good': ['Basic physics applications', 'Experimental setup', 'Data collection'],
        'needs_improvement': ['Physics principles', 'Measurement techniques', 'Basic analysis']
    },
    'chemistry_applications': {
        'excellent': ['Advanced chemical analysis', 'Reaction predictions', 'Laboratory techniques'],
        'good': ['Basic chemical reactions', 'Laboratory procedures', 'Data interpretation'],
        'needs_improvement': ['Chemical fundamentals', 'Laboratory safety', 'Basic procedures']
    },
    'content_strategy': {
        'excellent': ['Content planning', 'Audience targeting', 'Multi-platform strategy'],
        'good': ['Content calendars', 'Basic audience analysis', 'Platform considerations'],
        'needs_improvement': ['Content types', 'Basic planning', 'Platform awareness']
    },
    'curriculum_design': {
        'excellent': ['Curriculum development', 'Learning assessment', 'Educational theory'],
        'good': ['Lesson planning', 'Learning objectives', 'Teaching strategies'],
        'needs_improvement': ['Lesson structure', 'Learning goals', 'Basic pedagogy']
    },
    'education_planning': {
        'excellent': ['Educational program design', 'Assessment systems', 'Educational technology'],
        'good': ['Teaching plans', 'Basic assessment', 'Learning materials'],
        'needs_improvement': ['Lesson organization', 'Student engagement', 'Activity design']
    },
    'case_management': {
        'excellent': ['Comprehensive case planning', 'Resource coordination', 'Outcome assessment'],
        'good': ['Basic case management', 'Service referrals', 'Progress monitoring'],
        'needs_improvement': ['Client assessment', 'Documentation', 'Basic support planning']
    },
    'material_knowledge': {
        'excellent': ['Advanced material properties', 'Material innovation', 'Sustainability considerations'],
        'good': ['Material selection', 'Basic properties', 'Application techniques'],
        'needs_improvement': ['Material types', 'Basic characteristics', 'Common applications']
    },
    'animal_sciences': {
        'excellent': ['Animal physiology', 'Veterinary procedures', 'Behavior analysis'],
        'good': ['Animal anatomy', 'Basic healthcare', 'Behavior basics'],
        'needs_improvement': ['Animal types', 'Basic care', 'Observation skills']
    },
    'problem_solving': {
        'excellent': ['Complex problem analysis', 'Innovative solutions', 'Solution optimization'],
        'good': ['Problem breakdown', 'Solution identification', 'Implementation planning'],
        'needs_improvement': ['Problem recognition', 'Basic solutions', 'Logical approach']
    },
    'patient_care': {
        'excellent': ['Advanced care protocols', 'Patient assessment', 'Care planning'],
        'good': ['Basic patient care', 'Health monitoring', 'Care procedures'],
        'needs_improvement': ['Care fundamentals', 'Basic protocols', 'Patient communication']
    },
    'medical_knowledge': {
        'excellent': ['Advanced medical concepts', 'Diagnosis reasoning', 'Treatment planning'],
        'good': ['Basic medical terminology', 'Body systems', 'Common conditions'],
        'needs_improvement': ['Basic health concepts', 'Medical terms', 'Body awareness']
    }
}

activity_type_mapping = {
    'coding': ['text', 'code'],
    'computer_science': ['text', 'diagram'],
    'data_analysis': ['text', 'code', 'spreadsheet', 'chart'],
    'engineering_design': ['text', 'diagram', 'image', 'sketch'],
    'mathematics': ['text', 'calculation', 'equation'],
    'biology_experiments': ['text', 'diagram', 'image', 'report'],
    'healthcare': ['text', 'case_study', 'report'],
    'research_analysis': ['text', 'report', 'data'],
    'scientific_method': ['text', 'report', 'diagram'],
    'laboratory_procedures': ['text', 'procedure', 'image'],
    'business_case_studies': ['text', 'report', 'presentation'],
    'financial_analysis': ['text', 'spreadsheet', 'chart', 'report'],
    'market_research': ['text', 'survey', 'report', 'data'],
    'entrepreneurship': ['text', 'business_plan', 'presentation'],
    'management': ['text', 'plan', 'report'],
    'essay_writing': ['text', 'essay'],
    'critical_analysis': ['text', 'analysis'],
    'social_research': ['text', 'survey', 'report', 'data'],
    'creative_projects': ['text', 'image', 'design', 'multimedia'],
    'debate_argumentation': ['text', 'argument'],
    'visual_design': ['image', 'design', 'sketch'],
    'digital_illustration': ['image', 'design'],
    'user_research': ['text', 'report', 'diagram'],
    'prototyping': ['image', 'design', 'sketch'],
    'storytelling': ['text', 'script', 'storyboard'],
    'creative_writing': ['text', 'story', 'script'],
    'visual_composition': ['image', 'design', 'sketch'],
    'digital_editing': ['image', 'design', 'multimedia'],
    'spatial_planning': ['diagram', 'sketch', 'design'],
    'circuit_design': ['diagram', 'sketch', 'technical_drawing'],
    'electronics': ['diagram', 'technical_drawing', 'report'],
    'structural_design': ['diagram', 'technical_drawing', 'calculation'],
    'physics_applications': ['text', 'calculation', 'diagram', 'report'],
    'chemistry_applications': ['text', 'report', 'diagram', 'procedure'],
    'content_strategy': ['text', 'plan', 'content_calendar'],
    'curriculum_design': ['text', 'plan', 'lesson_plan'],
    'education_planning': ['text', 'plan', 'lesson_plan'],
    'case_management': ['text', 'case_study', 'plan'],
    'material_knowledge': ['text', 'report', 'image'],
    'animal_sciences': ['text', 'report', 'image'],
    'problem_solving': ['text', 'solution', 'diagram'],
    'patient_care': ['text', 'case_study', 'procedure'],
    'medical_knowledge': ['text', 'report', 'diagram']
}


# Pydantic models for requests and responses
class ClassLevel(BaseModel):
    level: str

class Scores(BaseModel):
    Numerical_Aptitude: float
    Spatial_Aptitude: float
    Perceptual_Aptitude: float
    Abstract_Reasoning: float
    Verbal_Reasoning: float

class PsychometricRequest(BaseModel):
    level: str
    categories: Optional[List[str]] = None
    questions_per_category: Optional[int] = 20

class PsychometricResponse(BaseModel):
    user_id: str
    responses: Dict[str, str]  # Maps question ID to answer (A, B, C, D)

class ActivitySubmission(BaseModel):
    activity_id: str
    response: str
    career_path: str
    class_level: str
    response_type: str = "text"
    image_data: Optional[str] = None

class GenerateActivityRequest(BaseModel):
    career_path: str
    class_level: str
    specific_area: Optional[str] = None

class CompleteAssessmentRequest(BaseModel):
    user_id: str
    orientation_responses: Dict[str, str]
    interest_responses: Dict[str, str]
    personality_responses: Dict[str, str]
    aptitude_scores: Scores

# Helper functions
def determine_stream(careers):
    stream_keywords = {
        'PCM': ['Engineer', 'Developer', 'Architect', 'Technical', 'Analyst', 'Programmer', 'Scientist', 'Data'],
        'PCB': ['Doctor', 'Biologist', 'Medical', 'Healthcare', 'Nurse', 'Pharmacist', 'Veterinarian', 'Therapist'],
        'Commerce': ['Accountant', 'Financial', 'Business', 'Marketing', 'Economics', 'Manager', 'Consultant', 'Entrepreneur'],
        'Humanities': ['Lawyer', 'Teacher', 'Writer', 'Social', 'Journalist', 'Historian', 'Psychologist', 'Counselor']
    }
    
    career_text = ' '.join(careers).lower()
    for stream, keywords in stream_keywords.items():
        if any(keyword.lower() in career_text for keyword in keywords):
            return stream
    return 'PCM'  # Default stream

# Generate aptitude questions as in the original code
def generate_questions_and_answers(topic: str, level: str, num_questions: int = 6) -> List[Dict]:
    prompt = f"""
    Generate {num_questions} multiple-choice questions based on the topic '{topic}' at a '{level}' difficulty level.
    Each question must have:
       "question": question text
       "options": array of 4 strings starting with A), B), C), D)
       "answer": one letter A, B, C, or D
       "explanation": brief explanation
    Return in JSON format."""
    
    response = model.generate_content(prompt)
    text = response.text.strip()
    
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    if not text.endswith("]"):
        text += "]"
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Provide a default set of questions if parsing fails
        return [
            {
                "question": f"Basic question about {topic}?",
                "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                "answer": "A",
                "explanation": "This is a default question due to parsing error."
            }
        ] * num_questions  # Repeat to get the desired number of questions

# Endpoints for different assessment types
@app.post("/generate_psychometric_assessment/")
async def generate_assessment(request: PsychometricRequest):
    if request.level not in ["10", "12"]:
        raise HTTPException(status_code=400, detail="Invalid level. Must be 10 or 12")
    
    try:
        # Default to all categories if none specified
        categories = request.categories or ["orientation", "interest", "personality", "aptitude"]
        questions_per_category = request.questions_per_category or 20
        
        # Validate categories
        valid_categories = ["orientation", "interest", "personality", "aptitude"]
        for category in categories:
            if category not in valid_categories:
                raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        
        # Initialize assessment
        assessment = {}
        
        # Get questions for each category
        if "orientation" in categories:
            assessment["orientation"] = random.sample(orientation_questions, min(questions_per_category, len(orientation_questions)))
            
        if "interest" in categories:
            assessment["interest"] = random.sample(interest_questions, min(questions_per_category, len(interest_questions)))
            
        if "personality" in categories:
            assessment["personality"] = random.sample(personality_questions, min(questions_per_category, len(personality_questions)))
            
        if "aptitude" in categories:
            # Use existing aptitude question generator
            aptitude_questions = []
            aptitude_categories = {
                "numerical_aptitude": "Math-based problem solving",
                "spatial_aptitude": "Shape manipulation and pattern recognition",
                "perceptual_aptitude": "Visual analysis and attention to detail",
                "abstract_reasoning": "Logical pattern identification",
                "verbal_reasoning": "Language-based comprehension"
            }
            
            questions_per_aptitude = questions_per_category // len(aptitude_categories)
            difficulty = "Basic" if request.level == "10" else "Intermediate"
            
            for trait, topic in aptitude_categories.items():
                questions = generate_questions_and_answers(topic, difficulty, questions_per_aptitude)
                for q in questions:
                    q["trait"] = trait
                    q["category"] = "aptitude"
                aptitude_questions.extend(questions)
                
            assessment["aptitude"] = aptitude_questions
        
        # Add question IDs for reference
        question_id = 1
        all_questions = []
        
        for category, questions in assessment.items():
            for i, question in enumerate(questions):
                question["id"] = f"{category}_{i+1}"
                question["question_number"] = question_id
                question_id += 1
                all_questions.append(question)
        
        return {
            "assessment_level": "Basic" if request.level == "10" else "Intermediate",
            "categories": categories,
            "total_questions": len(all_questions),
            "questions_by_category": assessment,
            "all_questions": all_questions
        }
        
    except Exception as e:
        print(f"Error generating psychometric assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Analyze orientation assessment responses
@app.post("/analyze_orientation_assessment/")
async def analyze_orientation(response: PsychometricResponse):
    try:
        # Parse responses
        answers = response.responses
        
        # Count responses for each trait value
        trait_counts = {}
        
        for question_id, answer in answers.items():
            if not question_id.startswith("orientation_"):
                continue
                
            # Find the question in the question bank
            matching_questions = [q for q in orientation_questions if q.get("id") == question_id]
            if not matching_questions:
                continue
                
            question = matching_questions[0]
            trait = question.get("trait")
            mapped_value = question.get("mapping", {}).get(answer)
            
            if trait and mapped_value:
                if trait not in trait_counts:
                    trait_counts[trait] = {}
                
                if mapped_value not in trait_counts[trait]:
                    trait_counts[trait][mapped_value] = 0
                
                trait_counts[trait][mapped_value] += 1
        
        # Determine dominant traits
        dominant_traits = {}
        for trait, values in trait_counts.items():
            if values:
                # Get the value with highest count
                dominant_value = max(values.items(), key=lambda x: x[1])[0]
                dominant_traits[trait] = dominant_value
        
        # Generate career matches based on dominant traits
        career_matches = []
        for trait, value in dominant_traits.items():
            if trait in career_trait_mappings["orientation"] and value in career_trait_mappings["orientation"][trait]:
                career_matches.extend(career_trait_mappings["orientation"][trait][value])
        
        # Count matches per career and sort
        career_counts = Counter(career_matches)
        top_careers = [career for career, count in career_counts.most_common(15)]
        
        # Determine stream and get recommendations
        stream = determine_stream(top_careers[:5])
        recommendations = career_recommendations[stream]
        
        # Generate report
        report = {
            "user_id": response.user_id,
            "assessment_type": "orientation",
            "trait_counts": trait_counts,
            "dominant_traits": dominant_traits,
            "career_matches": dict(career_counts.most_common(15)),
            "top_careers": top_careers,
            "stream": stream,
            "subject_recommendations": recommendations['subjects'],
            "skill_recommendations": recommendations['skills'],
            "report_summary": generate_orientation_report(dominant_traits, top_careers, stream)
        }
        
        return report
        
    except Exception as e:
        print(f"Error analyzing orientation assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Analyze interest assessment responses
@app.post("/analyze_interest_assessment/")
async def analyze_interest(response: PsychometricResponse):
    try:
        # Parse responses
        answers = response.responses
        
        # Count responses for each trait value
        trait_counts = {}
        
        for question_id, answer in answers.items():
            if not question_id.startswith("interest_"):
                continue
                
            # Find the question in the question bank
            matching_questions = [q for q in interest_questions if q.get("id") == question_id]
            if not matching_questions:
                continue
                
            question = matching_questions[0]
            trait = question.get("trait")
            mapped_value = question.get("mapping", {}).get(answer)
            
            if trait and mapped_value:
                if trait not in trait_counts:
                    trait_counts[trait] = {}
                
                if mapped_value not in trait_counts[trait]:
                    trait_counts[trait][mapped_value] = 0
                
                trait_counts[trait][mapped_value] += 1
        
        # Determine dominant traits
        dominant_traits = {}
        for trait, values in trait_counts.items():
            if values:
                # Get the value with highest count
                dominant_value = max(values.items(), key=lambda x: x[1])[0]
                dominant_traits[trait] = dominant_value
        
        # Generate career matches based on dominant traits
        career_matches = []
        for trait, value in dominant_traits.items():
            if trait in career_trait_mappings["interest"] and value in career_trait_mappings["interest"][trait]:
                career_matches.extend(career_trait_mappings["interest"][trait][value])
        
        # Count matches per career and sort
        career_counts = Counter(career_matches)
        top_careers = [career for career, count in career_counts.most_common(15)]
        
        # Determine stream and get recommendations
        stream = determine_stream(top_careers[:5])
        recommendations = career_recommendations[stream]
        
        # Generate report
        report = {
            "user_id": response.user_id,
            "assessment_type": "interest",
            "trait_counts": trait_counts,
            "dominant_traits": dominant_traits,
            "career_matches": dict(career_counts.most_common(15)),
            "top_careers": top_careers,
            "stream": stream,
            "subject_recommendations": recommendations['subjects'],
            "skill_recommendations": recommendations['skills'],
            "report_summary": generate_interest_report(dominant_traits, top_careers, stream)
        }
        
        return report
        
    except Exception as e:
        print(f"Error analyzing interest assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Analyze personality assessment responses
@app.post("/analyze_personality_assessment/")
async def analyze_personality(response: PsychometricResponse):
    try:
        # Parse responses
        answers = response.responses
        
        # Count responses for each trait value
        trait_counts = {}
        
        for question_id, answer in answers.items():
            if not question_id.startswith("personality_"):
                continue
                
            # Find the question in the question bank
            matching_questions = [q for q in personality_questions if q.get("id") == question_id]
            if not matching_questions:
                continue
                
            question = matching_questions[0]
            trait = question.get("trait")
            mapped_value = question.get("mapping", {}).get(answer)
            
            if trait and mapped_value:
                if trait not in trait_counts:
                    trait_counts[trait] = {}
                
                if mapped_value not in trait_counts[trait]:
                    trait_counts[trait][mapped_value] = 0
                
                trait_counts[trait][mapped_value] += 1
        
        # Determine dominant traits
        dominant_traits = {}
        for trait, values in trait_counts.items():
            if values:
                # Get the value with highest count
                dominant_value = max(values.items(), key=lambda x: x[1])[0]
                dominant_traits[trait] = dominant_value
        
        # Generate career matches based on dominant traits
        career_matches = []
        for trait, value in dominant_traits.items():
            if trait in career_trait_mappings["personality"] and value in career_trait_mappings["personality"][trait]:
                career_matches.extend(career_trait_mappings["personality"][trait][value])
        
        # Count matches per career and sort
        career_counts = Counter(career_matches)
        top_careers = [career for career, count in career_counts.most_common(15)]
        
        # Determine stream and get recommendations
        stream = determine_stream(top_careers[:5])
        recommendations = career_recommendations[stream]
        
        # Generate report
        report = {
            "user_id": response.user_id,
            "assessment_type": "personality",
            "trait_counts": trait_counts,
            "dominant_traits": dominant_traits,
            "career_matches": dict(career_counts.most_common(15)),
            "top_careers": top_careers,
            "stream": stream,
            "subject_recommendations": recommendations['subjects'],
            "skill_recommendations": recommendations['skills'],
            "report_summary": generate_personality_report(dominant_traits, top_careers, stream)
        }
        
        return report
        
    except Exception as e:
        print(f"Error analyzing personality assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Analyze aptitude assessment using the ML model
@app.post("/analyze_aptitude_assessment/")
async def analyze_aptitude(scores: Scores):
    try:
        # Convert scores to list format for ML model prediction
        scores_list = [
            scores.Numerical_Aptitude,
            scores.Spatial_Aptitude,
            scores.Perceptual_Aptitude,
            scores.Abstract_Reasoning,
            scores.Verbal_Reasoning
        ]
        
        # Use the ML model to predict career cluster
        cluster = kmeans.predict([scores_list])[0]
        careers = career_clusters[str(cluster)]
        
        # Determine aptitude levels for each score
        aptitude_levels = {}
        for attr, score in scores.dict().items():
            trait = attr.lower()
            if score >= 80:
                level = "high"
            elif score >= 50:
                level = "medium"
            else:
                level = "low"
            aptitude_levels[trait] = level
        
        # Generate additional career matches based on aptitude levels
        career_matches = list(careers)  # Start with ML model recommendations
        career_counts = Counter(career_matches)
        for trait, level in aptitude_levels.items():
            trait_name = trait.replace('_', '')
            if trait_name in career_trait_mappings["aptitude"] and level in career_trait_mappings["aptitude"][trait_name]:
                career_matches.extend(career_trait_mappings["aptitude"][trait_name][level])
                career_counts = Counter(career_matches)
        top_careers = [career for career, count in career_counts.most_common(15)]
        
        # Determine stream and get recommendations
        stream = determine_stream(careers)  # Use the ML model's career cluster for stream determination
        recommendations = career_recommendations[stream]
        
        # Generate report
        report = {
            "assessment_type": "aptitude",
            "aptitude_scores": scores.dict(),
            "aptitude_levels": aptitude_levels,
            "cluster": int(cluster),
            "ml_recommended_careers": careers,
            "career_matches": dict(career_counts.most_common(15)),
            "top_careers": top_careers,
            "stream": stream,
            "subject_recommendations": recommendations['subjects'],
            "skill_recommendations": recommendations['skills'],
            "report_summary": generate_aptitude_report(aptitude_levels, careers, stream)
        }
        
        return report
    
    except Exception as e:
        print(f"Error analyzing aptitude assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Comprehensive endpoint to analyze all assessment types together
@app.post("/analyze_complete_assessment/")
async def analyze_complete_assessment(request: CompleteAssessmentRequest):
    try:
        # Create individual response objects for each assessment type
        orientation_response = PsychometricResponse(
            user_id=request.user_id,
            responses=request.orientation_responses
        )
        
        interest_response = PsychometricResponse(
            user_id=request.user_id,
            responses=request.interest_responses
        )
        
        personality_response = PsychometricResponse(
            user_id=request.user_id,
            responses=request.personality_responses
        )
        
        # Get results from each assessment type
        orientation_results = await analyze_orientation(orientation_response)
        interest_results = await analyze_interest(interest_response)
        personality_results = await analyze_personality(personality_response)
        aptitude_results = await analyze_aptitude(request.aptitude_scores)
        
        # Collect all career matches from each assessment
        orientation_careers = set(orientation_results["top_careers"])
        interest_careers = set(interest_results["top_careers"])
        personality_careers = set(personality_results["top_careers"])
        aptitude_careers = set(aptitude_results["top_careers"])
        
        # Find common careers across all assessments
        common_careers = orientation_careers.intersection(interest_careers, personality_careers, aptitude_careers)
        
        # If not enough common careers, look for careers present in at least 3 assessments
        if len(common_careers) < 5:
            triple_matches = []
            for career in set().union(orientation_careers, interest_careers, personality_careers, aptitude_careers):
                count = 0
                if career in orientation_careers: count += 1
                if career in interest_careers: count += 1
                if career in personality_careers: count += 1
                if career in aptitude_careers: count += 1
                
                if count >= 3:
                    triple_matches.append(career)
            
            common_careers = set(triple_matches)
        
        # If still not enough, look for careers present in at least 2 assessments
        if len(common_careers) < 5:
            double_matches = []
            for career in set().union(orientation_careers, interest_careers, personality_careers, aptitude_careers):
                count = 0
                if career in orientation_careers: count += 1
                if career in interest_careers: count += 1
                if career in personality_careers: count += 1
                if career in aptitude_careers: count += 1
                
                if count >= 2:
                    double_matches.append(career)
            
            common_careers = set(double_matches)
        
        # Calculate a weighted score for each career
        career_scores = {}
        for career in common_careers:
            score = 0
            # Weight each assessment equally (max 25 points per assessment)
            orientation_rank = orientation_results["top_careers"].index(career) if career in orientation_careers else 100
            interest_rank = interest_results["top_careers"].index(career) if career in interest_careers else 100
            personality_rank = personality_results["top_careers"].index(career) if career in personality_careers else 100
            aptitude_rank = aptitude_results["top_careers"].index(career) if career in aptitude_careers else 100
            
            # Lower rank = higher score (up to 25 points per assessment)
            score += 25 - min(orientation_rank, 24) if orientation_rank < 100 else 0
            score += 25 - min(interest_rank, 24) if interest_rank < 100 else 0
            score += 25 - min(personality_rank, 24) if personality_rank < 100 else 0
            score += 25 - min(aptitude_rank, 24) if aptitude_rank < 100 else 0
            
            career_scores[career] = score
        
        # Sort careers by score
        top_careers = sorted(career_scores.items(), key=lambda x: x[1], reverse=True)
        top_career_list = [career for career, score in top_careers[:10]]
        
        # Determine stream and get recommendations based on top careers
        stream = determine_stream(top_career_list)
        recommendations = career_recommendations[stream]
        
        # Compile all trait information
        all_traits = {
            "orientation": orientation_results["dominant_traits"],
            "interest": interest_results["dominant_traits"],
            "personality": personality_results["dominant_traits"],
            "aptitude": aptitude_results["aptitude_levels"]
        }
        
        # Generate comprehensive report
        comprehensive_report = generate_comprehensive_report(
            all_traits,
            top_career_list,
            stream,
            orientation_results["top_careers"],
            interest_results["top_careers"],
            personality_results["top_careers"],
            aptitude_results["top_careers"]
        )
        
        # Compile final results
        report = {
            "user_id": request.user_id,
            "assessment_type": "comprehensive",
            "individual_results": {
                "orientation": orientation_results,
                "interest": interest_results,
                "personality": personality_results,
                "aptitude": aptitude_results
            },
            "combined_career_scores": dict(top_careers),
            "top_careers": top_career_list,
            "stream": stream,
            "subject_recommendations": recommendations['subjects'],
            "skill_recommendations": recommendations['skills'],
            "comprehensive_report": comprehensive_report
        }
        
        return report
    
    except Exception as e:
        print(f"Error analyzing complete assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions for report generation
def generate_orientation_report(dominant_traits, top_careers, stream):
    """Generate a detailed report based on orientation assessment results."""
    report = {
        "summary": "Your orientation assessment reveals your career values and work preferences.",
        "work_environment_preferences": "",
        "value_priorities": "",
        "work_style_preferences": "",
        "career_alignment": ""
    }
    
    # Work environment preferences
    if "structure_preference" in dominant_traits:
        structure = dominant_traits["structure_preference"]
        if structure == "structured":
            report["work_environment_preferences"] = "You thrive in structured, organized environments with clear expectations and procedures. You value stability and predictability in your work setting."
        elif structure == "flexible":
            report["work_environment_preferences"] = "You prefer flexible, adaptable work environments that allow for creativity and autonomy. You value freedom to determine your own approach to tasks."
        else:  # mixed
            report["work_environment_preferences"] = "You appreciate a balance between structure and flexibility in your work environment, adapting your approach based on the situation."
    
    # Value priorities
    if "value_priority" in dominant_traits:
        value = dominant_traits["value_priority"]
        if value == "security":
            report["value_priorities"] = "Job security and stability are important factors in your career decisions. You value reliable income and clear career advancement paths."
        elif value == "creativity":
            report["value_priorities"] = "Creative expression and innovation are central to your career satisfaction. You value opportunities to generate new ideas and approaches."
        elif value == "helping":
            report["value_priorities"] = "Making a positive impact and helping others is a core value for you. You're drawn to careers with clear social or environmental benefits."
        else:  # challenge
            report["value_priorities"] = "You're motivated by intellectual challenges and problem-solving opportunities. You value careers that push your abilities and offer continuous growth."
    
    # Work pace preferences
    if "work_pace" in dominant_traits:
        pace = dominant_traits["work_pace"]
        if pace == "steady":
            report["work_style_preferences"] = "You prefer a steady, methodical pace that allows you to focus on quality and attention to detail."
        elif pace == "fast":
            report["work_style_preferences"] = "You thrive in fast-paced, dynamic environments with variety and quick decision-making."
        else:  # varying
            report["work_style_preferences"] = "You adapt well to varying work rhythms, balancing periods of intense activity with more reflective phases."
    
    # Career alignment
    report["career_alignment"] = f"Based on your orientation profile, you show strong alignment with careers in {stream}. Top career matches include {', '.join(top_careers[:5])}."
    
    return report

def generate_interest_report(dominant_traits, top_careers, stream):
    """Generate a detailed report based on interest assessment results."""
    report = {
        "summary": "Your interest assessment reveals your career interests and passion areas.",
        "field_interests": "",
        "domain_preferences": "",
        "activity_affinities": "",
        "career_alignment": ""
    }
    
    # Field interests
    if "field_preference" in dominant_traits:
        field = dominant_traits["field_preference"]
        if field == "technical":
            report["field_interests"] = "You show strong interest in technical fields involving systems, data, or engineering. You enjoy working with technology and analytical problems."
        elif field == "creative":
            report["field_interests"] = "You're drawn to creative fields that involve design, expression, or innovation. You enjoy work that allows for artistic or original thinking."
        elif field == "helping":
            report["field_interests"] = "You're interested in careers that involve helping, teaching, or providing service to others. Personal impact is important to you."
        else:  # business
            report["field_interests"] = "You show interest in business-oriented fields involving management, strategy, or entrepreneurship. You enjoy organizational and commercial contexts."
    
    # Domain preferences
    if "domain_focus" in dominant_traits:
        domain = dominant_traits["domain_focus"]
        if domain == "humanities":
            report["domain_preferences"] = "You're drawn to humanities fields like literature, history, or social sciences. You enjoy exploring human experiences and cultural contexts."
        elif domain == "stem":
            report["domain_preferences"] = "Your interests align strongly with STEM (Science, Technology, Engineering, Math) domains. You enjoy working with data, systems, and scientific concepts."
        elif domain == "commerce":
            report["domain_preferences"] = "You show interest in commerce fields such as business, economics, or marketing. You enjoy understanding markets and commercial interactions."
        else:  # arts
            report["domain_preferences"] = "You're attracted to arts-related domains involving creative expression or design. You appreciate aesthetic qualities and creative processes."
    
    # Activity affinities
    if "activity_preference" in dominant_traits:
        activity = dominant_traits["activity_preference"]
        if activity == "analytical":
            report["activity_affinities"] = "You enjoy analytical activities involving research, calculation, or systematic problem-solving."
        elif activity == "hands_on":
            report["activity_affinities"] = "You prefer hands-on activities that involve physical skills, crafting, or direct manipulation of materials."
        elif activity == "creative":
            report["activity_affinities"] = "You're drawn to creative activities that involve generating new ideas, designs, or expressions."
        else:  # social
            report["activity_affinities"] = "You enjoy social activities that involve interaction, communication, or working directly with people."
    
    # Career alignment
    report["career_alignment"] = f"Based on your interest profile, you show strong alignment with careers in {stream}. Top career matches include {', '.join(top_careers[:5])}."
    
    return report

def generate_personality_report(dominant_traits, top_careers, stream):
    """Generate a detailed report based on personality assessment results."""
    report = {
        "summary": "Your personality assessment reveals your behavioral traits and work style.",
        "work_style_tendencies": "",
        "problem_solving_approach": "",
        "interpersonal_orientation": "",
        "leadership_tendencies": "",
        "career_alignment": ""
    }
    
    # Work style tendencies
    if "work_style" in dominant_traits:
        style = dominant_traits["work_style"]
        if style == "independent":
            report["work_style_tendencies"] = "You tend to work best independently with autonomy over your tasks and decisions. You're self-motivated and comfortable working with minimal supervision."
        elif style == "collaborative":
            report["work_style_tendencies"] = "You thrive in collaborative environments where you can work with others toward shared goals. You value team input and social interaction at work."
        else:  # mixed
            report["work_style_tendencies"] = "You adapt well to both independent and team-based work depending on the situation. You can be self-directed but also work effectively in groups."
    
    # Problem solving approach
    if "problem_approach" in dominant_traits:
        approach = dominant_traits["problem_approach"]
        if approach == "analytical":
            report["problem_solving_approach"] = "You approach problems analytically, using logical reasoning and systematic methods. You prefer evidence-based solutions and thorough analysis."
        elif approach == "creative":
            report["problem_solving_approach"] = "You tackle challenges creatively, looking for innovative or unconventional solutions. You think outside the box and generate multiple possibilities."
        else:  # pragmatic
            report["problem_solving_approach"] = "You take a pragmatic approach to solving problems, focusing on practical and efficient solutions. You value outcomes that work in the real world."
    
    # Interpersonal orientation
    if "social_orientation" in dominant_traits:
        orientation = dominant_traits["social_orientation"]
        if orientation == "people_focused":
            report["interpersonal_orientation"] = "You prioritize relationships and interpersonal dynamics in your work. You're attuned to others' needs and emotions and value positive social interactions."
        else:  # task_focused
            report["interpersonal_orientation"] = "You tend to be more focused on tasks and outcomes than interpersonal dynamics. You value efficiency and achieving objectives in your work."
    
    # Leadership tendencies
    if "leadership_style" in dominant_traits:
        leadership = dominant_traits["leadership_style"]
        if leadership == "directive":
            report["leadership_tendencies"] = "When leading, you tend to provide clear direction and structure. You're comfortable making decisions and setting expectations for others."
        elif leadership == "supportive":
            report["leadership_tendencies"] = "Your leadership style emphasizes support and development of team members. You focus on creating a positive environment and helping others succeed."
        else:  # delegative
            report["leadership_tendencies"] = "You prefer a delegative leadership approach that empowers team members with autonomy. You focus on assigning responsibilities based on strengths."
    
    # Career alignment
    report["career_alignment"] = f"Based on your personality profile, you show strong alignment with careers in {stream}. Top career matches include {', '.join(top_careers[:5])}."
    
    return report

def generate_aptitude_report(aptitude_levels, top_careers, stream):
    """Generate a detailed report based on aptitude assessment results."""
    report = {
        "summary": "Your aptitude assessment reveals your cognitive and logical skills.",
        "numerical_aptitude": "",
        "spatial_aptitude": "",
        "perceptual_aptitude": "",
        "abstract_reasoning": "",
        "verbal_reasoning": "",
        "career_alignment": ""
    }
    
    # Numerical aptitude
    if "numerical_aptitude" in aptitude_levels:
        level = aptitude_levels["numerical_aptitude"]
        if level == "high":
            report["numerical_aptitude"] = "You show strong numerical reasoning abilities, which is valuable for careers involving quantitative analysis, financial calculations, or mathematical modeling."
        elif level == "medium":
            report["numerical_aptitude"] = "You demonstrate solid numerical reasoning skills suitable for roles that require understanding and working with numerical data."
        else:  # low
            report["numerical_aptitude"] = "Your numerical reasoning shows room for development, which might be important if you're considering careers with significant quantitative components."
    
    # Spatial aptitude
    if "spatial_aptitude" in aptitude_levels:
        level = aptitude_levels["spatial_aptitude"]
        if level == "high":
            report["spatial_aptitude"] = "You excel in spatial reasoning, which is valuable for design, engineering, or visualization-heavy fields."
        elif level == "medium":
            report["spatial_aptitude"] = "You demonstrate solid spatial reasoning abilities suitable for roles involving moderate visual or spatial tasks."
        else:  # low
            report["spatial_aptitude"] = "Your spatial reasoning shows room for development, which might be relevant for fields requiring strong visual or spatial abilities."
    
    # Perceptual aptitude
    if "perceptual_aptitude" in aptitude_levels:
        level = aptitude_levels["perceptual_aptitude"]
        if level == "high":
            report["perceptual_aptitude"] = "You have strong perceptual abilities, allowing you to notice details and patterns quickly. This is valuable in quality control, editing, or diagnostic roles."
        elif level == "medium":
            report["perceptual_aptitude"] = "You show solid perceptual skills suitable for roles requiring moderate attention to visual details."
        else:  # low
            report["perceptual_aptitude"] = "Your perceptual aptitude shows room for growth, which could be relevant in fields requiring close attention to details."
    
    # Abstract reasoning
    if "abstract_reasoning" in aptitude_levels:
        level = aptitude_levels["abstract_reasoning"]
        if level == "high":
            report["abstract_reasoning"] = "You demonstrate excellent abstract reasoning, allowing you to understand complex concepts and identify patterns. This is valuable in research, programming, or strategic roles."
        elif level == "medium":
            report["abstract_reasoning"] = "You show solid abstract reasoning suitable for roles requiring conceptual thinking and pattern recognition."
        else:  # low
            report["abstract_reasoning"] = "Your abstract reasoning shows room for development, which might be relevant for fields requiring complex conceptual thinking."
    
    # Verbal reasoning
    if "verbal_reasoning" in aptitude_levels:
        level = aptitude_levels["verbal_reasoning"]
        if level == "high":
            report["verbal_reasoning"] = "You excel in verbal reasoning, which is valuable for communication-heavy roles in law, writing, teaching, or public relations."
        elif level == "medium":
            report["verbal_reasoning"] = "You demonstrate solid verbal reasoning suitable for roles requiring moderate communication and language skills."
        else:  # low
            report["verbal_reasoning"] = "Your verbal reasoning shows room for growth, which could be important in fields with heavy emphasis on language and communication."
    
    # Career alignment
    report["career_alignment"] = f"Based on your aptitude profile, you show strong alignment with careers in {stream}. Top career matches include {', '.join(top_careers[:5])}."
    
    return report

def generate_comprehensive_report(all_traits, top_careers, stream, orientation_careers, interest_careers, personality_careers, aptitude_careers):
    """Generate a comprehensive report that integrates all assessment results."""
    report = {
        "executive_summary": "Your comprehensive assessment reveals a holistic view of your career fit based on your values, interests, personality, and aptitudes.",
        "profile_overview": "",
        "career_match_analysis": "",
        "educational_pathway": "",
        "development_recommendations": ""
    }
    
    # Profile overview
    orientation_traits = all_traits["orientation"]
    interest_traits = all_traits["interest"]
    personality_traits = all_traits["personality"]
    aptitude_levels = all_traits["aptitude"]
    
    # Compile key traits into a profile summary
    profile_highlights = []
    
    # Add orientation highlights
    if "value_priority" in orientation_traits:
        if orientation_traits["value_priority"] == "security":
            profile_highlights.append("value stability and security")
        elif orientation_traits["value_priority"] == "creativity":
            profile_highlights.append("prioritize creative expression")
        elif orientation_traits["value_priority"] == "helping":
            profile_highlights.append("are motivated by helping others")
        elif orientation_traits["value_priority"] == "challenge":
            profile_highlights.append("seek intellectual challenges")
    
    # Add interest highlights
    if "field_preference" in interest_traits:
        if interest_traits["field_preference"] == "technical":
            profile_highlights.append("show interest in technical fields")
        elif interest_traits["field_preference"] == "creative":
            profile_highlights.append("are drawn to creative domains")
        elif interest_traits["field_preference"] == "helping":
            profile_highlights.append("are interested in service-oriented careers")
        elif interest_traits["field_preference"] == "business":
            profile_highlights.append("are attracted to business contexts")
    
    # Add personality highlights
    if "problem_approach" in personality_traits:
        if personality_traits["problem_approach"] == "analytical":
            profile_highlights.append("approach problems analytically")
        elif personality_traits["problem_approach"] == "creative":
            profile_highlights.append("take creative approaches to challenges")
        elif personality_traits["problem_approach"] == "pragmatic":
            profile_highlights.append("prefer practical problem-solving")
    
    # Add aptitude highlights
    high_aptitudes = [apt for apt, level in aptitude_levels.items() if level == "high"]
    if high_aptitudes:
        aptitude_str = ", ".join([apt.replace("_aptitude", "").replace("_reasoning", " reasoning") for apt in high_aptitudes])
        profile_highlights.append(f"show strong aptitude in {aptitude_str}")
    
    # Combine highlights into profile overview
    report["profile_overview"] = "You " + ", ".join(profile_highlights) + ". This profile suggests alignment with careers that combine these elements."
    
    # Career match analysis
    common_matches = set(top_careers).intersection(
        orientation_careers[:10], 
        interest_careers[:10], 
        personality_careers[:10], 
        aptitude_careers[:10]
    )
    
    if common_matches:
        common_matches_str = ", ".join(list(common_matches)[:5])
        report["career_match_analysis"] = f"Several career paths show strong alignment across all assessment areas, including {common_matches_str}. These careers leverage your unique combination of values, interests, personality traits, and aptitudes."
    else:
        report["career_match_analysis"] = f"Your top recommended careers represent an optimal balance of your values, interests, personality traits, and aptitudes. While some careers may align more strongly with specific aspects of your profile, the recommendations aim to provide holistic fit."
    
    # Educational pathway
    report["educational_pathway"] = f"Based on your assessment results, an educational pathway focused on {stream} subjects would align well with your profile. Core subjects to consider include {', '.join(career_recommendations[stream]['subjects']['core'])}, with potential electives in {', '.join(career_recommendations[stream]['subjects']['electives'][:3])}."
    
    # Development recommendations
    technical_skills = career_recommendations[stream]['skills']['technical'][:3]
    soft_skills = career_recommendations[stream]['skills']['soft'][:3]
    
    report["development_recommendations"] = f"To prepare for your recommended career paths, focus on developing technical skills like {', '.join(technical_skills)}, along with soft skills including {', '.join(soft_skills)}. Consider both formal education and practical experiences to build your competencies in these areas."
    
    return report


app.post("/generate_questions/")
async def generate_questions(class_input: ClassLevel):
    if class_input.level not in ["10", "12"]:
        raise HTTPException(status_code=400, detail="Invalid class level. Must be 10 or 12")
    
    try:
        difficulty = "Basic" if class_input.level == "10" else "Intermediate"
        categories = {
            "Numerical Aptitude": "Math-based problem solving",
            "Spatial Aptitude": "Shape manipulation and pattern recognition",
            "Perceptual Aptitude": "Visual analysis and attention to detail",
            "Abstract Reasoning": "Logical pattern identification",
            "Verbal Reasoning": "Language-based comprehension"
        }
        
        all_questions = []
        for category, topic in categories.items():
            questions_list = generate_questions_and_answers(topic, difficulty)
            for q in questions_list:
                q["Category"] = category
                all_questions.append(q)
        
        return {"questions": all_questions}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_careers/")
async def predict_careers(scores: Scores):
    try:
        scores_list = [
            scores.Numerical_Aptitude,
            scores.Spatial_Aptitude,
            scores.Perceptual_Aptitude,
            scores.Abstract_Reasoning,
            scores.Verbal_Reasoning
        ]
        
        cluster = kmeans.predict([scores_list])[0]
        careers = career_clusters[str(cluster)]
        stream = determine_stream(careers)
        recommendations = career_recommendations[stream]
        
        return {
            "cluster": int(cluster),
            "recommended_careers": careers,
            "stream": stream,
            "subject_recommendations": recommendations['subjects'],
            "skill_recommendations": recommendations['skills'],
            "input_scores": scores.dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_activities/")
async def generate_activities(request: GenerateActivityRequest):
    if request.class_level not in ["10", "12"]:
        raise HTTPException(status_code=400, detail="Invalid class level. Must be 10 or 12")
    
    try:
        # Find relevant activity areas for this career
        activity_areas = []
        
        # If a specific area is requested and it's valid, use only that
        if request.specific_area and request.specific_area in skill_gap_assessment:
            activity_areas = [request.specific_area]
        else:
            # Otherwise get recommended areas based on career
            if request.career_path in career_activity_mapping:
                activity_areas = career_activity_mapping[request.career_path]
            else:
                # If career not in mapping, get stream and use its activity areas
                stream = determine_stream([request.career_path])
                activity_areas = career_recommendations[stream]['activity_areas'][:3]  # Get top 3 activity areas
        
        difficulty = "Basic" if request.class_level == "10" else "Intermediate"
        
        # Generate activities for each relevant area
        activities = []
        for i, area in enumerate(activity_areas):
            activity = generate_career_activity(request.career_path, area, difficulty)
            activity["id"] = f"{request.career_path.replace(' ', '_').lower()}_{area}_{i}"
            activity["activity_area"] = area
            
            # Add recommended submission formats
            if area in activity_type_mapping:
                activity["recommended_formats"] = activity_type_mapping[area]
            else:
                activity["recommended_formats"] = ["text"]
            
            activities.append(activity)
        
        return {"activities": activities, "career_path": request.career_path}
    
    except Exception as e:
        print(f"Error in generate_activities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate_activity/")
async def evaluate_activity(submission: ActivitySubmission):
    try:
        # Extract activity area from activity_id more safely
        try:
            parts = submission.activity_id.split('_')
            # Default to a valid activity area if we can't extract one
            activity_area = parts[-2] if len(parts) >= 2 else "coding"
            
            # Verify the activity area is valid, if not use a fallback
            if activity_area not in skill_gap_assessment:
                activity_area = "coding"  # Default fallback
        except:
            activity_area = "coding"  # Fail-safe default
        
        # Prepare the content for evaluation based on response type
        content_for_evaluation = submission.response
        
        # Handle different submission types
        if submission.response_type == "image" and submission.image_data:
            # For image submissions, we'll include a note about the image and the text description
            content_for_evaluation = f"[Image submission] {submission.response}"
            
        
        # Evaluate the response with error handling
        try:
            evaluation = evaluate_activity_response(
                activity_area=activity_area,
                response=content_for_evaluation,
                career_path=submission.career_path,
                class_level=submission.class_level,
                response_type=submission.response_type
            )
        except Exception as eval_error:
            # If evaluation fails, use a default structure
            print(f"Evaluation error: {str(eval_error)}")
            evaluation = default_evaluation()
        
        # Get overall score safely
        try:
            overall_score = evaluation.get("overall", {}).get("score", 75)
        except:
            overall_score = 75  # Default score if we can't extract it
            
        # Determine skill level
        skill_level = determine_skill_level(overall_score)
        
        # Get skill development recommendations
        try:
            skill_development = get_skill_development_recommendations(activity_area, skill_level)
        except Exception as skill_error:
            # Default skill development if it fails
            print(f"Skill development error: {str(skill_error)}")
            skill_development = {
                "focus_areas": ["Core skills practice", "Guided learning", "Project-based practice"],
                "learning_plan": default_learning_plan()
            }
        
        # Add submission metadata
        submission_meta = {
            "timestamp": datetime.now().isoformat(),
            "activity_area": activity_area,
            "response_type": submission.response_type,
            "class_level": submission.class_level
        }
        
        return {
            "evaluation": evaluation,
            "skill_level": skill_level,
            "skill_development": skill_development,
            "submission_meta": submission_meta
        }
    
    except Exception as e:
        print(f"Evaluation endpoint error: {str(e)}")
        # Return a graceful error response instead of failing
        return JSONResponse(
            status_code=200,  # Return 200 with error info instead of 500
            content={
                "evaluation": {
                    "overall": {"score": 70, "feedback": "Your submission has been processed."}
                },
                "skill_level": "good",
                "skill_development": {
                    "focus_areas": ["Continue practicing in this area"],
                    "learning_plan": {"resources": ["Online tutorials and practice"]}
                },
                "error_info": "There was an issue processing your submission. The results shown are provisional."
            }
        )

def generate_career_activity(career: str, activity_area: str, difficulty: str) -> Dict:
    """Generate a career-specific activity for assessment."""
    activity_prompts = {
        'coding': "Create a program or algorithm to solve a problem",
        'computer_science': "Explain a computing concept or algorithm",
        'data_analysis': "Analyze data and draw conclusions",
        'engineering_design': "Design a solution to an engineering problem",
        'mathematics': "Solve a mathematical problem",
        'biology_experiments': "Design or explain a biology experiment",
        'healthcare': "Address a healthcare scenario",
        'research_analysis': "Analyze research findings",
        'scientific_method': "Apply the scientific method",
        'laboratory_procedures': "Describe laboratory procedures",
        'business_case_studies': "Analyze a business case",
        'financial_analysis': "Analyze financial data",
        'market_research': "Design a market research plan",
        'entrepreneurship': "Develop a business concept",
        'management': "Address a management challenge",
        'essay_writing': "Write an essay on a topic",
        'critical_analysis': "Analyze a text or argument",
        'social_research': "Design a social research study",
        'creative_projects': "Create a creative project",
        'debate_argumentation': "Develop arguments for a debate",
        'visual_design': "Create a visual design",
        'digital_illustration': "Create a digital illustration",
        'user_research': "Design a user research plan",
        'prototyping': "Create a prototype design",
        'storytelling': "Develop a story or narrative",
        'creative_writing': "Write a creative piece",
        'visual_composition': "Create a visual composition",
        'digital_editing': "Edit a digital asset",
        'spatial_planning': "Create a spatial plan",
        'circuit_design': "Design a basic circuit",
        'electronics': "Explain or design an electronic component",
        'structural_design': "Design a structural element",
        'physics_applications': "Apply physics principles to solve a problem",
        'chemistry_applications': "Apply chemistry principles in a scenario",
        'content_strategy': "Develop a content strategy",
        'curriculum_design': "Design a learning curriculum",
        'education_planning': "Create an educational plan",
        'case_management': "Manage a fictional case scenario",
        'material_knowledge': "Analyze or select materials for a purpose",
        'animal_sciences': "Address an animal science scenario",
        'problem_solving': "Solve a complex problem",
        'patient_care': "Develop a patient care plan",
        'medical_knowledge': "Apply medical knowledge to a case"
    }
    
    base_prompt = activity_prompts.get(activity_area, "Complete a task related to your field")
    
    difficulty_level = "basic concepts" if difficulty == "Basic" else "more advanced concepts"
    
    # Determine appropriate submission formats
    submission_formats = activity_type_mapping.get(activity_area, ["text"])
    format_guidance = ", ".join(submission_formats)
    
    prompt = f"""
    Generate a detailed {difficulty} level activity for a student interested in becoming a {career}.
    This activity should focus on {activity_area.replace('_', ' ')} and involve {difficulty_level}.
    
    The activity should be appropriate for submission as: {format_guidance}.
    
    Include in your response:
    1. A clear title
    2. Detailed instructions
    3. Required resources or materials
    4. Expected outcome or deliverable
    5. Time estimation (appropriate for a class {difficulty} student)
    6. Learning objectives (what skills this develops)
    7. Submission format guidelines (how the student should submit their work)
    
    Return as a JSON object with these exact fields.
    """
    
    response = model.generate_content(prompt)
    text = response.text.strip()
    
    # Clean up the response to ensure valid JSON
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    try:
        # Find the JSON part of the response if mixed with text
        json_start = text.find('{')
        json_end = text.rfind('}')
        
        if json_start != -1 and json_end != -1:
            json_text = text[json_start:json_end+1]
            activity = json.loads(json_text)
        else:
            activity = json.loads(text)
        
        # Add career and activity area information
        activity["career"] = career
        activity["activity_area"] = activity_area
        return activity
    except json.JSONDecodeError:
        # If JSON parsing fails, create a structured response
        return {
            "title": f"{career} {activity_area.replace('_', ' ')} Activity",
            "instructions": "Complete the activity according to your understanding of the field.",
            "resources": "Basic tools related to the field",
            "expected_outcome": "A completed project demonstrating your understanding",
            "time_estimation": "1-2 hours",
            "learning_objectives": f"Develop skills in {activity_area.replace('_', ' ')}",
            "submission_format_guidelines": f"Submit your work as {', '.join(submission_formats)}",
            "career": career,
            "activity_area": activity_area
        }

def evaluate_activity_response(activity_area: str, response: str, career_path: str, class_level: str, response_type: str = "text") -> Dict:
    """Evaluate a student's response to an activity."""
    difficulty = "Basic" if class_level == "10" else "Intermediate"
    
    # Ensure safe inputs for the prompt
    safe_activity_area = activity_area.replace('_', ' ')
    safe_class_level = class_level if class_level in ["10", "12"] else "12"
    
    prompt = f"""
    You are evaluating a {safe_class_level}th grade student's response to an activity in the area of {safe_activity_area} 
    related to their interest in becoming a {career_path}. The response was submitted as a {response_type}.
    
    Student's response:
    '''
    {response}
    '''
    
    Please evaluate this response on the following criteria:
    1. Understanding of concepts
    2. Application of knowledge
    3. Creativity and originality
    4. Technical correctness
    5. Communication clarity
    
    For each criterion, provide a score out of 20 and brief feedback.
    Also provide an overall score out of 100 and general feedback with strengths and areas for improvement.
    
    Return your evaluation as a JSON object with exactly this structure:
    {{
      "understanding_of_concepts": {{ "score": (number), "feedback": (string) }},
      "application_of_knowledge": {{ "score": (number), "feedback": (string) }},
      "creativity_and_originality": {{ "score": (number), "feedback": (string) }},
      "technical_correctness": {{ "score": (number), "feedback": (string) }},
      "communication_clarity": {{ "score": (number), "feedback": (string) }},
      "overall": {{ 
        "score": (number), 
        "feedback": (string),
        "strengths": (string),
        "areas_for_improvement": (string)
      }}
    }}
    
    The response MUST be valid JSON without comments or extra text.
    """
    
    try:
        generation_response = model.generate_content(prompt)
        text = generation_response.text.strip()
        
        # Clean up the response to ensure valid JSON
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # Additional cleanup to handle potential JSON issues
        # Remove any surrounding text that might prevent JSON parsing
        json_start = text.find('{')
        json_end = text.rfind('}')
        
        if json_start != -1 and json_end != -1:
            text = text[json_start:json_end+1]
        
        try:
            evaluation = json.loads(text)
            
            # Validate the expected structure exists, fill in missing parts if needed
            required_criteria = [
                "understanding_of_concepts", 
                "application_of_knowledge", 
                "creativity_and_originality",
                "technical_correctness", 
                "communication_clarity", 
                "overall"
            ]
            
            # Create the structure if any top-level keys are missing
            for criterion in required_criteria:
                if criterion not in evaluation:
                    if criterion == "overall":
                        evaluation[criterion] = {
                            "score": 75,
                            "feedback": "Good work overall.",
                            "strengths": "Shows competence in this area.",
                            "areas_for_improvement": "Continue practicing to develop skills further."
                        }
                    else:
                        evaluation[criterion] = {
                            "score": 15,
                            "feedback": f"Demonstrates adequate {criterion.replace('_', ' ')}."
                        }
            
            # Ensure overall contains all required fields
            if "overall" in evaluation:
                required_overall = ["score", "feedback", "strengths", "areas_for_improvement"]
                for field in required_overall:
                    if field not in evaluation["overall"]:
                        default_values = {
                            "score": 75,
                            "feedback": "Good work overall.",
                            "strengths": "Shows competence in this area.",
                            "areas_for_improvement": "Continue practicing to develop skills further."
                        }
                        evaluation["overall"][field] = default_values[field]
            
            return evaluation
            
        except json.JSONDecodeError:
            # If JSON parsing fails, create a structured evaluation
            return default_evaluation()
            
    except Exception as e:
        print(f"Error in evaluate_activity_response: {str(e)}")
        return default_evaluation()

def default_evaluation() -> Dict:
    """Return a default evaluation structure if the AI evaluation fails."""
    return {
        "understanding_of_concepts": {"score": 15, "feedback": "Shows basic understanding"},
        "application_of_knowledge": {"score": 15, "feedback": "Adequate application of knowledge"},
        "creativity_and_originality": {"score": 15, "feedback": "Some creative elements present"},
        "technical_correctness": {"score": 15, "feedback": "Generally technically sound"},
        "communication_clarity": {"score": 15, "feedback": "Communication is clear"},
        "overall": {
            "score": 75,
            "feedback": "Good work overall. Continue practicing to improve skills in this area.",
            "strengths": "Basic understanding and application",
            "areas_for_improvement": "Could develop more depth and creativity"
        }
    }

def determine_skill_level(score: int) -> str:
    """Determine skill level based on evaluation score."""
    if score >= 85:
        return "excellent"
    elif score >= 70:
        return "good"
    else:
        return "needs_improvement"

def get_skill_development_recommendations(activity_area: str, skill_level: str) -> Dict:
    """Get skill development recommendations based on activity area and skill level."""
    # Validate inputs
    if activity_area not in skill_gap_assessment:
        activity_area = "coding"  # Default to coding if activity_area is invalid
        
    valid_skill_levels = ["excellent", "good", "needs_improvement"]
    if skill_level not in valid_skill_levels:
        skill_level = "good"  # Default to good if skill_level is invalid
    
    # Get recommendations from the skill gap assessment mapping
    if activity_area in skill_gap_assessment and skill_level in skill_gap_assessment[activity_area]:
        recommendations = skill_gap_assessment[activity_area][skill_level]
    else:
        # Default recommendations if mapping not found
        recommendations = [
            "Fundamental skills practice",
            "Guided projects",
            "Structured learning resources"
        ]
    
    # Generate personalized learning plan
    prompt = f"""
    Create a personalized learning plan for a student who shows a {skill_level} level in {activity_area.replace('_', ' ')}.
    
    Focus areas should include:
    {', '.join(recommendations)}
    
    The plan should include:
    1. Specific learning resources (books, courses, websites)
    2. Practice activities
    3. Progress tracking metrics
    4. Timeline suggestions
    
    Return as a JSON object with exactly this structure:
    {{
      "resources": [list of resources],
      "practice_activities": [list of activities],
      "progress_metrics": [list of metrics],
      "timeline": string or object with timeline information
    }}
    
    The response MUST be valid JSON without comments or extra text.
    """
    
    try:
        generation_response = model.generate_content(prompt)
        text = generation_response.text.strip()
        
        # Clean up the response to ensure valid JSON
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # Additional cleanup to handle potential JSON issues
        json_start = text.find('{')
        json_end = text.rfind('}')
        
        if json_start != -1 and json_end != -1:
            text = text[json_start:json_end+1]
        
        try:
            learning_plan = json.loads(text)
            
            # Ensure the required structure exists
            required_fields = ["resources", "practice_activities", "progress_metrics", "timeline"]
            for field in required_fields:
                if field not in learning_plan:
                    learning_plan[field] = default_learning_plan()[field]
                
            return {
                "focus_areas": recommendations,
                "learning_plan": learning_plan
            }
        except json.JSONDecodeError:
            # If JSON parsing fails, use default structure
            return {
                "focus_areas": recommendations,
                "learning_plan": default_learning_plan()
            }
    except Exception as e:
        print(f"Error in get_skill_development_recommendations: {str(e)}")
        return {
            "focus_areas": recommendations,
            "learning_plan": default_learning_plan()
        }

def default_learning_plan() -> Dict:
    """Return a default learning plan structure."""
    return {
        "resources": ["Recommended textbooks", "Online courses", "Tutorial websites"],
        "practice_activities": ["Guided exercises", "Projects", "Peer collaboration"],
        "progress_metrics": ["Self-assessment quizzes", "Project completion", "Portfolio development"],
        "timeline": "3-6 months of consistent practice"
    }

# Add a new endpoint to get activity suggestions by career
@app.get("/career_activities/{career_path}")
async def get_career_activities(career_path: str):
    try:
        if career_path in career_activity_mapping:
            activity_areas = career_activity_mapping[career_path]
        else:
            # If career not in mapping, get stream and use its activity areas
            stream = determine_stream([career_path])
            activity_areas = career_recommendations[stream]['activity_areas']
            
        activities_info = {}
        for area in activity_areas:
            if area in skill_gap_assessment:
                activities_info[area] = {
                    "name": area.replace('_', ' ').title(),
                    "submission_formats": activity_type_mapping.get(area, ["text"]),
                    "skill_areas": skill_gap_assessment[area]["good"]  # Use "good" level as reference
                }
        
        return {
            "career_path": career_path,
            "recommended_activity_areas": activity_areas,
            "activities_info": activities_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add an endpoint to submit and store activities for portfolio building
class PortfolioSubmission(BaseModel):
    user_id: str
    activity_id: str
    career_path: str
    response: str
    response_type: str = "text"
    evaluation: Optional[Dict] = None
    