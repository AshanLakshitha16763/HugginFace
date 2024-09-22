from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMchain, OpenAI



load_dotenv(find_dotenv()) 


# Image convert into text des

def imagetext(url):

    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=0)
    text = image_to_text(url)[0]["generated_text"]
    print(text)

    return text

imagetext("./bear.jpg")


#LLM part

def generate_des(scenario):
    template= """
        you are the story teller;
        you can generate the story based on the image, the story should be no more than 200 wordf;

        context:{scenario}
        STORY: 
        
        """

    prompt = PromptTemplate(template=template, input_variable=["scenario"])

    story_llm = LLMchain(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True)
    story= story_llm.predict(scenario=scenario)
    print(story)
    return story






#Text convert into speech