from typing import Optional

from fastapi import FastAPI
from starlette.responses import RedirectResponse

from datalib import Datastore, Fields
from fastapi.responses import FileResponse, HTMLResponse
from fastapi import Request

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


datastore = Datastore().load_from_dir("data/sample42/")
articles = {}
for article in datastore.articles.to_dict(orient='records'):
    articles[article[Fields.article_id]] = article

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/info")
def get_info():
    return {"info": datastore.info()}

@app.get("/random_user")
def random_user():
    customer_id = list(datastore.customers.sample(1)[Fields.customer_id])[0]
    customer_info = datastore.customers[datastore.customers[Fields.customer_id] == customer_id].to_dict(orient='records')
    transactions = datastore.transactions[datastore.transactions[Fields.customer_id] == customer_id]
    transactions = transactions.to_dict(orient='records')
    for transaction in transactions:
        transaction["article_detail"] = articles[transaction['article_id']]
    return {"customer_id":customer_info, "transactions": transactions}

@app.get("/image/{article_id}", response_class=FileResponse)
async def image(article_id : int):
    article_id_to_str = "0" + str(article_id)
    return f"data/small_images_01/images/{article_id_to_str[:3]}/{article_id_to_str}.jpg"

@app.get("/article/{id}", response_class=HTMLResponse)
async def get_transactions(request: Request, id: str):
    article_detail = articles[int(id)]
    return templates.TemplateResponse("article.html", {"request": request,
                                                       "article": article_detail})

@app.get("/random/", response_class=HTMLResponse)
async def get_transactions(request: Request):
    customer_id = list(datastore.customers.sample(1)[Fields.customer_id])[0]
    return RedirectResponse(url=f"/transactions/{customer_id}")

@app.get("/transactions/{id}", response_class=HTMLResponse)
async def get_transactions(request: Request, id: str):
    customer_id = id
    # customer_id = list(datastore.customers.sample(1)[Fields.customer_id])[0]
    customer_info = datastore.customers[datastore.customers[Fields.customer_id] == customer_id].to_dict(
        orient='records')[0]
    customer_info = {str(k): str(v) for k, v in customer_info.items()}
    transactions = datastore.transactions[datastore.transactions[Fields.customer_id] == customer_id]
    transactions = transactions.to_dict(orient='records')
    d2s = lambda x: f"{x.day}/{x.month}/{x.year}"
    for transaction in transactions:
        transaction["article_detail"] = articles[transaction['article_id']]
        transaction["timestamp"] = d2s(transaction["t_dat"])

    return templates.TemplateResponse("item.html", {"request": request,
                                                    "transactions": transactions,
                                                    "customer":customer_info})
