### Web service deployment 

To run the model locally using a web interface add the following dependencies to the current project conda environment:

```shell
    $ pip install django django-cors-headers pydot   
```

Then start it with the following command run on the base project path:

```shell
    $ python web/manage.py runserver --insecure
```

Open the url http://127.0.0.1:8000/index.html to browse the service web page.



