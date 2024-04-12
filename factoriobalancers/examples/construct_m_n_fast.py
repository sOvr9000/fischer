

from fischer.factoriobalancers import BeltGraph
from math import gcd
from functools import lru_cache

_1ns = {
    "3": "eJwzsDbSMbU2sDJUMLQy0jFWMLYy1TEEAC1SBBA=",
    "5": "eJwzsDbSMbE2sDJUMLQy0jFWMLYy0TFVMLUy1jEEAEmBBTE=",
    "7": "eJwzsDbSMdExtzawMlQwtDLSMVYwtjLRMVUwtTLXMQQAVJoFmA==",
    "9": "eJwFwUkNAAAIAzArE7AHR4Aw/PuitQs290wOVzCRKjpKzcFoWQ96kQbJ",
    "11": "eJwFwUkNAAAIAzArE7AHN8nw74vWLlicMzlcwUSq2GgNF6umP3oRBr8=",
    "13": "eJwFwUkNAAAIAzArE7AHN8nw74vWLlicMzlcwUSq2GgNF6ukP3oLBr0=",
    "15": "eJwFwUcRADAMAzAqBuBHsy8Of16V3jmTzb0ng8kZCCULpeZgtLQPiHoHKA==",
    "17": "eJwNxjkRADAMAzAqBuAh77Vx+PNqNck2WLxrcriCiVSx0Tp0HF0O5r8fptwH8g==",
    "19": "eJwzsDbSMdGxsDawMlQwtDLSMVYwtjLRMVUwtTLXMVQwt7LQsVSwBLKNAabaB/A=",
    "21": "eJwFwcERADAIArBVGIBHsfVU3H+vJmeDj01pjwU5eHH9mEgXhXJzMJZYH8eRCK0=",
    "23": "eJwFwUkBACAMAzArFdAHYxfr/PsiOXsZLL49MpguHa5gIlVstB4Ho6Z9t2kIVQ==",
    "25": "eJwNyTEBACAMAzArFdCDDsZY8e8L8mbc4OLmucOCHJyYXkykNwvlw0Zb4m+5mA8AgAnU",
    "27": "eJwFwUkBACAMAzArFdAHYxfr/PsiOXsZLL49MpguHa5gIlVstB4Ho6R9t2MIUw==",
    "29": "eJwFwckBACAIA7BVOkAfcihQ9t/L5Kwz+dh7ZDA5A6HkxdVjodQcjIL2AbddCFE=",
    "31": "eJwFwUkBACAQAzErFTAPCixH8e+LpL3OZHGwX4vldIZGJqXKYmvncHVj4w/ZqQkO",
    "33": "eJwNycERACAIA7BVOkAfFA+Fuv9emm/iJjebihsW5OTCclEobx4cNwdjBfVfHtYDE7oKMA==",
    "35": "eJwNycERACAIA7BVOkAfFPWQsv9emm9ikpuXigkLcnJhefPguCiULxttBfVfbtYDE0AKLg==",
    "37": "eJwNycERACAIA7BVOkAfFEXOsv9e+swlJrl5qJiwICcXljcL5cNG+1K4VlD/9dkPEuAKKg==",
    "39": "eJwNyDEBACAMAzArFdCDDgZb8e8Lcmbc4OJm3WFBDk5MLybSmwfHxUZb+iW5qQcAognW",
    "41": "eJwNycERACAIA7BVOkAfVAWOuv9e+swl7uJhUXHDgry4sX2YSBcb7aEwVlD/9ZkPEt4KKA==",
    "43": "eJwNycERACAIA7BVOkAfFESPuv9e+swlbnJxU3HDgpwslBcb7c2D46EwVlD/9VkPEtwKJg==",
    "45": "eJwVysERACAIA7BVOkAfVESPuv9e6juJMzi5qKDyhAV5MJGeLJQXN7abQvuf57KSfQE8wgrn",
    "47": "eJwNyTkBADAMAzEqAXBD3D8uf16tVuVtDBYH5U0r5EaP7sGM6cWO7UNFWYn+y4UeKJAKjw==",
    "49": "eJwVysERACAIA7BVOkAfVOTQuv9e6juJMzi5qKDyhAV5MJGeLJSbQntxY/uf57KSfQE9Kgrp",
    "51": "eJwNyTEBACAMAzArFdCDDsZY8e8L8mbc4OLmucOCHJyYXkykNwvlw0Zb+iG5qAcAjAnS",
    "53": "eJwNyDEBACAMAzArFdCDbgxG8e8Lcmbc4ORi32FBDibSk4Xy4sZ28+BY+iU5qQcAkAnQ",
    "55": "eJwNybkBACAMA7FVMsAVMV/A7L8XqFXexmCxUd60Qm706B7MmF5UlDcnjpXov1zoASiKCo0=",
    "57": "eJwVysERACAIA7BVOkAfFETPuv9e6juJkxycVFB5woKcLJQHG+3JheVNYfuf57KSKqjcrAuaIQxi",
    "59": "eJwNybkBACAMA7FVMsAVMRAes/9eoFZ5G4PJRnnTCrnRo3tQUZ6sWN6cOFai/3KhByiECos=",
    "61": "eJwNybkBACAMA7FVMsAVMeE1++8FapW30ZlslDetkBsV5c6I4cmK5c2JYyX6Lxd6KH4KiQ==",
    "63": "eJwVyrkBwDAMA7FVNMAVouWX3n+vODWQt9GZbJSoblohNyrKnRHDkxXLmxPH/3kuq9AHVAcLSg==",
    "65": "eJwVyLERwCAMBMFWVMAHnAQjePrvy2bDHTc1tUXeYQKnKspTK5ZbRHvrxDH8BSZFBfWiP3doC94=",
    "67": "eJwNyckRACAMA7FWUoAfWcIxMf33Bfoq79DUFnnTBB6qKE+tWN46cdwi2qT4j6l/lEH9AHZJC9g=",
    "69": "eJwVyrkRACAMA7BVPICLOLnwmP33AmopTnJwUUHlCQtyslBuCu3BienFje1/nstKqqCyxL6n4QyZ",
    "71": "eJwVyLkBwDAMA7FVNAALn+SX3n+vOCXQbqpraou8zQROVZS7RgxPrVjeOnEMD2BSVFB/8AGQewxB",
    "73": "eJwVyrENACAMA7BXckAG0lJQw/9/AaMljxOcXFRQdYYFOZhITxbKixvbTaEtPUj+N6G0in0Bm6YMdA==",
    "75": "eJwVyrkRACAMA7BVPICLOOG5mP33AmopTnJwUUHlCQtyslAenJhe3NhuCu1/nstKqqCyxL6mrQyV",
    "77": "eJwVyckBACAMArBVGICHtF7F/fdSv0k7wc7JTcVpFuRgIt05MDy5sLxZKEt8LSuohPJDXZAXDEM=",
    "79": "eJwNybkBACAMA7FVMoCLHOE1++8FrZS3qWtqi7xpAjdVlLtGDE+tWN46cUyK/5j6ShnEA4/MDD0=",
    "81": "eJwVyrERACEMxMBWXMAFyMYwHP339Xyq1bipqSVS9B0mcKqiPNXRXtqxfV45Bj3G/1tBmRZ8p/0MmQ==",
    "83": "eJwVyLkBwDAMA7FVNAALn+SX3n+vOCjRbqpraou8zQROVZS7RgxPrVjeOnEMr8CkqKD+4AOQVww/",
    "85": "eJwVycEBABAMA8BVMkAe0qLE/nvhe9dOsHNyUXGaBTmYSHcODE8Wyosb2xJfywoqofxQF5AVDEE=",
    "87": "eJwNybkBACAMA7FVMoCLHOE1++8FapW3qWtqi7xpAjdVlLtGDE+tWN46cUyK/5j6RxnEA4+6DDs=",
    "89": "eJwVyckBACAMArBVGICHtNYD999L/SbtBDsHFxWnWZCDiXRnoTw4Mb24sS3xtaygEsoPdQGQEww/",
    "91": "eJwVybkBwDAQArBVGIDCHH7x/nvFaaV2i52Tm6rbIihFw+kcGJlcWNk8OJH4WlFRhvyDP5ARDD0=",
    "93": "eJwVyskBACAMArBVGICHtNYD999LfSftBDsHFxVUnWZBDibSnYXy4MT04sa2xMfyvwmlVZQuwtkNAg==",
    "95": "eJwVyskBACAMArBVGIBHaT1x/73UdxIn2Ti4qKDyhAU5WSg3dnQPTkwvbmz/81xWUgWVJeoCwbUM+A==",
    "97": "eJwVyrERACAMw8BVPICLOAFymP33Alq94iQHFxVUnrAgJwvlwYnpxUZ7U9j+z3NZSRVUr/cFmjEMag==",
    "99": "eJwNysERACAIA7BVOkAfFlSw7r+XfnMZNzi52XdYkIOJ9OTC8mah3Dw4lj5JVv6gdFEPVcYLVQ==",
    "101": "eJwVyrERACAMw8BVPICLOCFwmP33Alq94iQHJxVUnrAgJwvlwUZ7cmF5U9j+z3NZSRVUr/cFmi8MaA==",
    "103": "eJwVyrkRADEQAsFUCABjWfSUUP55nc6dnrrNwUUV1bciKE3DGZyYWdzYORRO/ue5oqYM+XV/mi0MZg==",
    "105": "eJwVysERACEIA8BWUkAeBESH2H9fd7534yYXNxVUUn3DgpwslBcb7c2D46Ewfu13+eWCymrOB82CDSs=",
    "107": "eJwNyjEBACAMAzArFdBjZQxY8e8L7iTu4OTioeKGBXkwkZ4slBc3tg8bbQX1XVb+oXRTD4OCDBA=",
    "109": "eJwNybkBACAQArBVGIDiEF/cfy9tk7qNnZObqlsRlEbD6RwYmVxY2Tw4UVH/FfmrHFMPg5AMDg==",
    "111": "eJwVyjkBwDAMA0AqAqDBsvNZ4c+r6XwXNzm4eKig8oYFOVkoD05ML25sHzba/3kuK6mCyk19tNAMzw==",
    "113": "eJwVi8kNADAIw1ZhgDwIh1DT/fcq/Vmy7TdQoIMB1nXRqEBaqtDWGtBGZ/noZ+upH6cxxX3b2GJiHi4CDno=",
    "115": "eJwNyjEBACAMAzArFdBjZYxB8e8L7iTu4OTipuKGBXkwkZ4slBcb7c2DYwX1XVb+oXRTD4N8DA4=",
    "117": "eJwNybkBACAQArBVGIDiEF/cfy9Nm7qNnZObqlsRlEbD6RwYmVxY2Tw4UVH/FfmfHFMPg4AMDA==",
    "119": "eJwVyrEBwCAMA7BXfICHOAEC5v+/SmcpbnJwcVNB5Q0LcrJQHpyYXmy0Nw+O//NcVlIFlZv6ALTKDM0=",
    "121": "eJwVysEBwCAMw8BVPIAfOCEEzP57lX51Gjc4ubipQcUdFuRgIj1ZKC822psHx//zXFZQCaVVfKHcrA8h3Q5U",
    "123": "eJwVyrkBwCAMA8BVNIAKy8Y8Yv+9Quq7uMnByU0FlTcsyMlCebDRnlxY3jw4/s9zWUkVVG7qA7TEDMs=",
    "125": "eJwVyskBACAMArBVGIBHaT1x/73UdxIn2Ti4qKDyhAU5WSg3dnQPTkwvbmz/81xWUgWVi7q0vgzJ",
    "127": "eJwVyrkBwCAMA8BVNIAKy8Y8Yv+9Euq7uMnByU0FlVTfsCAnC+XBRntyYXnz4Pi13+WXCyqrqQ/ptg2O",
    "129": "eJwVisERACAIw1ZhgD6ogB51/73EV3NJ/S4kNuhgXheNWggLJcpKG8eOerj1P9MpBmZDTLCM9UU/EvQOIg==",
    "131": "eJwVyckBACEMA7FWUoAfDCEcpv++lv1K7XYNTW0xbjOBuzLSQxXlqRXLWyeOQa8x+ZA0Q1RQP8AHB6MN+w==",
    "133": "eJwVysEVACEIA9FWUkAORuAhsf++dr3On3U3k4daVN5lQd4MhJOFclNoHw7G7/ldVrChsJIqqF6YDxOGDig=",
    "135": "eJwVy8EVACEIA9FWUkAOREQesf++dj3Pn7iLm4cKalN9w4K8mEhvFsqHjfZQGD/2d1nJgdJvKqispvIDTcQO6Q==",
    "137": "eJwdyrkRwDAMA7BVOAAL05L80PvvFV1aHMabTC4eKt+wIE8GwslCeXFj+/DiWmqSrGC3sJJqqB/0AQedDfk=",
    "139": "eJwNybkBwDAMA7FVNACLnGT5offfK0aL76aGprbI+5nAqYryUEd7asXy1olj0GtMigrK9FvalOAHBuwN9w==",
    "141": "eJwVyrkBwCAMA8BVNIAKy8Y8Yv+9Euq7uMnJTQWV1LhhQU4Wyk2hPbmwvHlw/Nrv8ssFlTWohtoq9gdNgw7j",
    "143": "eJwViscBACAMAldhAB5iEgvuv5f6vNJOZ3JwUY3K0yzInYFwslAenJhe3Nj+z+uy4rHCSqqg+kIXMVwOiQ==",
    "145": "eJwVysEVACEIA9FWUkAOROChsf++dr3On7iLxU0FVTcsyIuJdLHRHgrjzYPj9/wuKzlQWkU11C/kBxOADiI=",
    "147": "eJwVy7ERACAMw8BVPIALTJILmP33Amq9xplMLmpQSfUZFuTJQDhZKDeF9uLG9mevywo2FP5TQWU9FhdOVg7r",
    "149": "eJwdirkNADEMw1bRACqsOM6j7L/XGVcRBBlvcHLxUPHCgjyYSE8Wyosb24cX1wqqu6xkM63qR/W7PgY7DfE=",
    "151": "eJwNyrkBwDAMA7FVNACLnGT5offfK66B76aGprbI+5nAqYryUEd7asXy1olj0GNMigrK9Ju0KcEPBsQN8w==",
    "153": "eJwVy7ERACEMA8FWVIAChG0Mov++ng9vdm7cyeTippLqOyzIk4FwslBebLQ3D44lPpYVLxX+n4LKakofMzkOlQ==",
    "155": "eJwVyrkBwDAMA7FVNACLnCX5offfK0kNPHeoNLXFEHUfE3goI13qaE+tWN46cQz6GP83gzQlOmiTghdAJQ62",
    "157": "eJwdiskRACEMw1pxAX5gknCY/vta2JdmJLXTmRxcVKPyNAtyZyCcLJQHJ6YXN7bfc7us4GVYSRVUv9AHP34Osg==",
    "159": "eJwVyrkBwCAMA8BVNIAKy8Y8Yv+9Quq7uMnByU0FlTcsyMlCebDRnlxY3jw4/s9zWUkVVFY/UFtFfTCVDoU=",
    "161": "eJwVyrkRwCAQA8BWVIACdM8ciP77sol31w0WN7WooOouC3IwkS422kNhvHlw/Nrv8ssJpVVUQ20l5wNMsQ7h",
    "163": "eJwVyrkRwDAMA7BVOAAL05L80PvvFaXFYbzJ5OKh8g0L8mQgnCyUFze2Dy+upSbJCnYLK6mG+iE++ZcNyg==",
    "165": "eJwVyrkRwCAQA8BWVIACdA/Mif77sol31w0WN7WooOouC3IwkS422psHx0Nh/Nrv8ssJpVVUQ20l5wNL+w7f",
    "167": "eJwVyrkBACAMw8BVPIALTBIes/9eQHtSO53JwUU1Kk+zIHcGwslCeXBienFj+z+vy4qnCiupguqDLjE4Doc=",
    "169": "eJwVisERACAIw1ZhgD6oiErdfy/xlcslfgcmFg7o10WjBsJCE2mphW1bB2UlOtidYqAZYvbJ/F4P+DENxg==",
    "171": "eJwVyckRwDAMw8BWWAAfomX5oPvvK8oLg9l4g5OLh4oXFuTBRHqyUF7c2D68uFZQ7bKS3bSqVfV/fvhPDcQ=",
    "173": "eJwVisERACAIw1ZhgD6ogErdfy/1lbskfgYSExt0MI+LRg2EhRJlpYllSxttrf+8TjHwGGKCZawv+gIw1A6J",
    "175": "eJwVyrkBwCAMA8BVNIAKy8Y8Yv+9Quq7uMnByU0FlTcsyMlCebDRnlxY3jw4/s9zWUkVVFY/VltFfTCDDoM=",
    "177": "eJwVy8EBwCAMw8BVPIAfmCQEzP57lb51GncyubippPoOC/JkIJwslBcb7c2DY+kFyQo+Fv6fgspqKj4y2Q6R",
    "179": "eJwVissVACAIw1ZhgB6ogJ+6/17iLS+J34HExAYdzOuiUQNhoURZaWLZ0saxo/90pxjdGGKCDfUFHzEUDoU=",
    "181": "eJwVisERACAIw1ZhgD6ogGjdfy/1lbskfgYSEwt0MI+LRg2EhRJlpYm21sK2rf+8TjHwGGKCZawv+gIw0g6H",
    "183": "eJwVyrkBwCAMA8BVNIAKy8Y8Yv+9Quq7uMnByU0FlTcsyMlCebDRnlxY3jw4/s9zWUkVVFa/pLaK+gAwcQ6B",
    "185": "eJwVirkRACAMw1bxAC4wSXjM/nsBle4ktdOZHFxUo/I0C3JnIJwslAcnphc3tv/zuqzgY1hJFVRf1AUw0A6F",
    "187": "eJwVirkRACAMw1bxAC4wSXjM/nsBle4ktdOZHFxUo/I0C3JnIJwslAcnphc3tv/zuqzgY1hJFVRfxAUwzg6D",
    "189": "eJwVi7EBACAIw17hgA5UQLT+/5c6ZUjiZyAxsUAHE+zjolEDYaFEWWmirbWwbetnz1MMPIb+VMYSG4wLbSsPTA==",
    "191": "eJwVyrcBwCAQA8BVNIAK9IEg9t/Lpr4bN1ic3NSggqo7LMjBRLrYaE8uLG8eHL/2u/xyQmkV1VBbSX1r8g9C",
    "193": "eJwVi8kRACAIA1uhgDyIgAyx/77U7x5+FhIbdDDBPi4atRAWSpSVNtpaA9roZ89TjEcZ+lMZS2zMBT6/Drw=",
    "195": "eJwVyrkRwCAQA8BWVIACdA/Mif77sol31w0WN7WooOouC3IwkS422psHx0Nh/Nrv8ssJpVVUQ22J8wFL9Q7d",
    "197": "eJwVyrEBwCAMA7BXfIAHnBAg5v+/Smdp3ODk4qHiDgtyMJGeLJQXN7YPG22Jj2UFlVBa9abqh/4A+MQNyA==",
    "199": "eJwVyckBACAMArBVGICHtNYD999LzTftBDsHFxWnWZCDiXRnoTw4Mb24sS3xtaygEkqr3qp+5AX44g3G",
    "201": "eJwVyrcBwCAQA8BVNIAK9IEg9t/Lpr4bN1ic3FRQdYcFOZhIFxvtyYXlzYNjiT/L7yaUVlEN9ZPzATFxDos=",
    "203": "eJwVysERACEIA8BWUkAeRGCU2H9fd7534y4WDxXUouqGBXkxkS422pvC9uFg/Nrv8ssJpVVUQ+3h/gA95Q62",
    "205": "eJwVyrEBwCAMA7BXfIAHnBAC5v+/Smdp3ODk4qbiDgtyMJGeLJQXG+3Ng2OJj2UFlVBa9abqh/4A+MINxg==",
    "207": "eJwVyrkBwCAMA8BVNIAKy8Y8Yv+9Quq7uMnByU0FlTcsyMlCebDRnlxY3jw4/s9zWUkVVFY/VluiPjB9DoE=",
    "209": "eJwVyrEBwCAMA7BXfIAHTBIC5v+/WmZp3Mnk4qYmlXdYkCcD4WShvNhobx4cS/xZfjegsJIqqJ70BzFvDok=",
    "211": "eJwNyrkRADEMA7FWVACDW8nyQ/ff1znF4Lupoakt8n4mcKqiPNTRnlqxvHXiGB6BSVFBmX6T9hI/620NmQ==",
    "213": "eJwVyrcBwCAQA8BVNIAK9IEg9t/Lpr4bN1ic3FRQdYcFOZhIFxvtyYXlzYNjiT/L7yaUVlEN9ZP+ADFtDoc=",
    "215": "eJwVyrcBwCAQA8BVNIAK9IEg9t/Lpr4bN1ic3FRQdYcFOZhIFxvtyYXlzYNjiT/L7yaUVlEN9ZP8ADFrDoU=",
    "217": "eJwVy8EBwCAMw8BVPIAfOCEEzP57tbx1Gjc4ubipoCbVd1iQg4n0ZKG82GhvHhxL/LP8eELpNxVUVlP6AG3WD04=",
    "219": "eJwVyrkBwCAMA8BVNIAKy8Y8Yv+9Quq7uMnByU0FlTcsyMlCebDRnlxY3jw4/s9zWUkVVFa/qrZEfTBZDn0=",
    "221": "eJwVyskRACAQArBWKICH7KEj9t+X+k7GCRYnNaig6gwLcjCRLjbakwvLm8L2b8/lnxNKq6iG+oW+PS8OsA==",
    "223": "eJwVyrcBwCAQA8BVNIAK9IEg9t/Lpr4bN1ic3NSggqo7LMjBRLrYaE8uLG8eHL/2u/xyQmkV1VBboj5r7A9A",
    "225": "eJwVy8EVACEIA9FWUkAOBESesf++dj3Pn7jJxU0FldSi5oYFOVkoLzbam4PxoXD85N/l5wsqv6uhtobnA3ojD3c=",
    "227": "eJwNyrEBACAIA7BXOKADFQGt//8lc+JvYaNwQH8uGrUQFtpISxXaWgfXrujgOMUYY4g5k6kGP+q0DZU=",
    "229": "eJwzsDbSMbY2sDJUMLQCsgAXxwLv",
    "231": "eJwVyrEBwCAMA7BXfIAHnBAg5v+/Smdp3ODk4qEGFXdYkIOJ9GShvLixfdho/+e5rKASSqteUrmpDyH/DlY=",
    "235": "eJwVyrEBwCAMA7BXfIAHnBAg5v+/Smdp3ODk4qEGFXdYkIOJ9GShvLixfdho/+e5rKASSqteVbmpDyHvDlQ=",
    "237": "eJwVyrkBwCAMA8BVNIAKy8Y8Yv+9Quq7uMnByU0FlTcsyMlCebDRnlxY3jw4/s9zWUkVVFY/VruoDyH9DlI=",
    "239": "eJwVyrcBwCAQA8BVNIAK9IEg9t/Lpr4bN1ic3NSggqo7LMjBRLrYaE8uLG8eHL/2u/xyQmkV1VD7UB9cvw8X",
    "241": "eJwzsDbSMbY2sDJUMLQCsgAXxwLv",
    "243": "eJwVyrEBwCAMA7BXfIAHnBAC5v+/Smdp3ODk4qYGFXdYkIOJ9GShvNhobx4c/+e5rKASSqteVbmpDyHpDlI=",
    "245": "eJwVyrkBwCAMA8BVNIAKy8Y8Yv+9Quq7uMnByU0FlTcsyMlCebDRnlxY3jw4/s9zWUkVVFa/pHZRHyHtDlA=",
    "247": "eJwVyrEBwCAMA7BXfIAHTBIC5v+/WmZp3Mnk4qYGNam8w4I8GQgnC+XFRnvz4Pi13+WXAworqYLKTX1cuQ8V",
    "249": "eJwVyrEBwCAMA7BXfIAHTBIC7v9/FWZpfJPJxU0NalL5DQvyZCCcLJQXG+3Ng+PXrssvBxRWUgWV1bzSbtYP3BsQoA==",
    "251": "eJwVyrcBwCAQA8BVNIAK9IEg9t/Lpr4bN1ic3NSggqo7LMjBRLrYaE8uLG8eHL/2u/xyQmkV1VC7qQ9csw8T",
    "253": "eJwVyrcBwCAQA8BVNIAK9IEg9t/Lpr4bN1ic3NSggqo7LMjBRLrYaE8uLG8eHL/2u/xyQmkV1VA7qQ9crQ8R",
    "255": "eJwVy8EBwCAMw8BVPIAfmBACZv+92rx1Gm9ycfNQg5rUouoNC/JkILyYSG8WyocX1y3/LrcPKNxXQmkV9QGbRg/a",
    "279": "eJwVi7kBwCAMA1fRACoijDEW++8VaO/5zuDk4qYGtag+nwV5MBCeTKQXC+XNRlvi1fLLAworqUvS7y6orKbyBwE7ERI=",
    "285": "eJwVy8EBwCAMw8BVPIAfmDQJmP33Krx1GmeyuKhBTeqj6gwL8mQgnBTSxUZ7cWP7ydvl5wMKvyuhtIpqqK1k/g8/ETU=",
    "287": "eJwdysEBACEIA8FWKCAPAwIa++/rTr+zO45jorDAATpYZ4hGOcJCE2mpQltrYdvW3f5O3TmMIeaTFAtsYz/5AP5jEQA=",
    "313": "eJwdi8ERACEIA1uhgDyMCEjsv6/Te2VmdzPOxEJigwNMsM8QjZpwcy2EhRJlpY221suup+i462L8IPTeZSyxQf8AAGgRDA==",
    "315": "eJwVi7EBwCAMw17xAR4wIQm4//9VmDRIGt/kYnFTg1pUfcOCPBkILybSxUZ78+D4ZdfLCl6G35RQWkU11FZS8QP/EREG",
    "317": "eJwdysEBACEIA8FWKCAPAwIa++/r9L6zO45jorDAATpYZ4hGOcJCE2mpQltrYdvW226n3hzGEBMXUiywjf1DfP5kEQI=",
    "319": "eJwdyrEBwCAMA7BXfIAHTBIC7v9/tXSWxjOZXNzUoCaVz7AgTwbCyUJ5sdHePDi+7XP55oDCSqqgsvqntop6Af3cEPw=",
    "329": "eJwVy8EBwCAMw8BVPIAfmBACZv+92rx1Gm9y8VCDmtSi9hsW5MlAeDGRLgrlw4vrln+X2wcU7iuhtDZVUFnJ+gAOQxEz",
    "333": "eJwVi7EBACAIw17hgA5WBKX+/5c4JmnHnVhIHHCACdYdolETbq6FsFBi29ZBWenPulP0tnQx0Bz65xZbLDAe7/YQ4w==",
    "335": "eJwVysEBwCAMw8BVPIAfmDQJmP33Knx1GmfyY3FRg5pUnWFBngyEPybSxUZ7cWP7bdflNwcUVl5QWkU11C/oB+3yENc=",
    "337": "eJwVy7ERACEMA8FWVICCF8YYi/77eghvbvY7g5OLm5rUovp8FuTBQHgykV4slDcbbYl3y4qbCj+WUPrhgspqSj/xrhDp",
    "341": "eJwVy8EBwCAMw8BVPIAfmJAEzP57tbx1GndysbipSS2q7rAgTwbCi4l0sdHePDiW+Gf58YDCb0ooraIaaispff/gEQg=",
    "343": "eJwVi7kBwCAMA1fRACoQxjEW++8VaO8ZZ3JxU4Na1Ef1GRbkyUB4MZEuCuXNRvuV18sKFhR+X0Lpd19SVlPxAxDkET8=",
    "345": "eJwVjMkRACAIA1tJAXkYERlj/30p7z3GnVzc1KAmtah9hwV5MhBeTKQ3C+VD4bjNz+X2Awp3lVBa/1NQWcnzAA1jETE=",
    "349": "eJwdysEBACEIA8FWKCAPAwIa++/r9L6zO45jorDAATpYZ4hGOcJCE2mpQltrYdvW226n3hzGEBMXUiywjf0DP/5iEQA=",
    "351": "eJwVyrEBwCAMA7BXfIAHTBIC7v9/FWZpfJPJxU0NalL5DQvyZCCcLJQXG+3Ng+PXrssvBxRWUgWV1XeoraJ+7SUQ0w==",
    "357": "eJwVi8ERACAIw1ZhgD6siEjdfy/xmUsy7sTCxgEHuMG6QzRqws21EBbaSEsdlJV+1p6it6OLgebQn9OYYoHxAO/AEOE=",
    "359": "eJwVisERACAIw1ZhgD4sCGjdfy/1l0syjmOisMABOlhniEY5wkITaalCW2th29bfXqf+HMYQ82WmWOCD/oIX7c4Q1Q==",
    "365": "eJwVysEBwCAMw8BVPIAfmDQJmP33KnxPGmfyY3FRg5pUnWFBngyEPybSxUZ7cWP7bbfLbw4orOSFtIpqqB/sH+1qENc=",
    "367": "eJwVyrEBwCAMA7BXfIAHTBIC7v9/FWZpfJPJxU0NalL5DQvyZCCcLJQXG+3Ng+PXrssvBxRWUgWV1fepraJ+7RMQ0Q==",
    "369": "eJwVi8ERACAIw1ZhgD6siEjdfy/xmUsy7sTCxgEHuMG6QzRqws21EBbaSEsdlJV+1p6id0EXA82hP6cxxQLjAe+KEN8=",
    "371": "eJwVysEBwCAMw8BVPIAfmDQJmP33KnylG2fyY3FRg5pUnWFBngyEPybSxUZ7cWP7sfvlhwMKKy9SWkU11C/oB+2qENM=",
    "373": "eJwVysEBwCAMw8BVPIAfmDQJmP33KnxPGmfyY3FRg5pUnWFBngyEPybSxUZ7cWP7bbfLbw4orOSFtIpqqB/0D+1oENU=",
    "375": "eJwVyrEBwCAMA7BXfIAHTBIC7v9/FWZpfJPJxU0NalL5DQvyZCCcLJQXG+3Ng+PXrssvBxRWUgWV1XerraJ+7QEQzw==",
    "377": "eJwVysEBwCAMw8BVPIAfmDQJmP33KnxPGmfyY3FRg5pUnWFBngyEPybSxUZ7cWP7bbfLbw4orOSFtIpqqB/kD+1mENM=",
    "379": "eJwVysEBwCAMw8BVPIAfmDQJmP33KnxPGmfyY3FRg5pUnWFBngyEPybSxUZ7cWP7bbfLbw4orOSFtIpqqB/ED+1kENE=",
    "381": "eJwVi7kBwCAMA1fRACoQjjEW++8VaO8ZZ/Lj4qYGNalF9RkW5MlA+GMivVgobzbar7xefn1AYSUvSL+7oLKayh8zlxGe",
    "383": "eJwVy8EBwCAMw8BVPIAfmJAEzP57tbx1GndysbipQU1qUXWHBXkyEF5MpIuN9ubB8ZN/l58PKPyuhNIqqqG2kvoAMkkRlA==",
    "385": "eJwVy8EBwCAMw8BVPIAfmBACZv+92rx1Gm9y8VCDmtSi9hsW5MlAeDGRLgrlw4vrln+X2wcU7iuhtDZVUFnB+gAOPREx",
    "391": "eJwVi7kBACAIA1dhgBREBDTuv5fa3uNnYKKwQAcnWMdFowbCQhNpqUJba2Hb1s+epxiPMvSnNKZY4CP9DS/upxDX",
    "401": "eJwVi7kBwCAMA1fRACoijDEW++8VaO/5zuDk4qYGtag+nwV5MBCeTKQXC+XNRlvi1fLLAworL1f6zQWV1ZR+8LUQ5Q==",
    "403": "eJwVy8EBwCAMw8BVPIAfmJAEzP57tbx1GndysbipSS2q7rAgTwbCi4l0sdHePDiW+Gf58YDCb0ooraIaaisoff/YEQY=",
    "405": "eJwVjMkRACAIA1tJAXkYERlj/30p7z3GnVzc1KAmtah9hwV5MhBeTKQ3C+VD4bjNz+X2Awp3lVBa/1NQWcHzAA1dES8=",
    "409": "eJwdi8EBACEIw1ZhgD6oCGjdf6/Te+WRxM/ARGGBDk6wjotGDYSFJtJShbbWwratl11PMXAZelMaUyywjf0bfv8HEQI=",
    "413": "eJwVy8EBwCAMw8BVPIAfNSGEmP33Kryl+87g5OKmJrWoPp8FeTAQnkykFwvlzUZbukGygncLP5ZQ+uGCymoqfvEYEOU=",
    "415": "eJwVyrEBwCAMA7BXfIAHTBIC7v9/FWZpfJPJxU0NalL5DQvyZCCcLJQXG+3Ng+PXrssvBxRWUgWV1XeoraB+7R8Q0Q==",
    "419": "eJwVy8ERACAIA8FWKCAPIgoa++9L/GVmc34HJhIbdHCCeV00aiAsNLFsKVFW2jh29G/tFKONoR/1WGKCZawvfO5xENU=",
    "425": "eJwVi7EBwCAMw17xAR4wIQm4//9VmDRIGt/kYnFTg1pUfcOCPBkILybSxUZ78+D4ZdfLCl6G35RQWkU11M+cH+4NENk=",
    "427": "eJwVy8EBwCAMw8BVPIAfmBACZv+92rx1Gm9y8VCDmtSi9hsW5MlAeDGRLgrlw4vrln+X2wcU7iuhtDZVUPmyPv0SEQY=",
    "431": "eJwVyrEBwCAMA7BXfIAHTBIC7v9/FWZpfJPJxU0NalL5DQvyZCCcLJQXG+3Ng+PXrssvBxRWUgWV1fepraB+7Q0Qzw==",
    "433": "eJwVi7EBwCAMw17xAR4wIQm4//9VmDRIGt/kYnFTg1pUfcOCPBkILybSxUZ78+D4ZdfLCl6G35RQWkU11M/0D+4LENc=",
    "435": "eJwVjMkRACAIA1tJAXkYERlj/30p7z3GnVzc1KAmtah9hwV5MhBeTKQ3C+VD4bjNz+X2Awp3lVBa/1NQWeJ5DVcRLQ==",
    "437": "eJwVi7EBwCAMw17xAR4wIQm4//9VmDRIGt/kYnFTg1pUfcOCPBkILybSxUZ78+D4ZdfLCl6G35RQWkU11M/kD+4JENU=",
    "439": "eJwVi7EBwCAMw17xAR4wIQm4//9VmDRIGt/kYnFTg1pUfcOCPBkILybSxUZ78+D4ZdfLCl6G35RQWkU11M/ED+4HENM=",
    "441": "eJwVi7EBwCAMw17xAR4wIYS4//9VmDRIGt/k4uahBrWoTfU3LMiTgfBiIr1ZKB822q+8XlbwMvy+hNLvLqispuIHNEgRoA==",
    "443": "eJwVy8EBwCAMw8BVPIAfmCQE3P33Krx1Gt9kcnFTg5pUfsOCPBkIJwvlxUZ78+D4sdvlhwMKK6mCyur7qK2gfuzpEMs=",
    "447": "eJwVy8EBwCAMw8BVPIAfmJAEzP57tbx1GndysbipQU1qUXWHBXkyEF5MpIuN9ubB8ZN/l58PKPyuhNIqqqG2gvoAMkMRkg==",
    "449": "eJwVy8EBwCAMw8BVPIAfmJAEzP57tbx1GndysbipSS2q7rAgTwbCi4l0sdHePDiW+Gf58YDCb0ooraIa6kfOB+7UENs=",
    "457": "eJwVy8EBwCAMw8BVPIAfmJAEzP57tbx1GndysbipSS2q7rAgTwbCi4l0sdHePDiW+Gf58YDCb0ooraIa6kf6A+7SENk=",
    "461": "eJwVy8EBwCAMw8BVPIAfmJAEzP57tbx1GndysbipSS2q7rAgTwbCi4l0sdHePDiW+Gf58YDCb0ooraIa6kfyA+7QENc=",
    "463": "eJwVy8EBwCAMw8BVPIAfmJAEzP57tbx1GndysbipSS2q7rAgTwbCi4l0sdHePDiW+Gf58YDCb0ooraIa6kfiA+7OENU=",
    "465": "eJwVy8ERwCAMA7BVPIAfmBBCzP57tbx1GndycfNQk1rUpvoOC/JkILyYSG8WyoeNtsSf5TcCCr+XUPrtgspqSh81HRGi",
    "471": "eJwVyrEBwCAMA7BXfIAHTBIC7v9/FWZpfJPJxU0NalL5DQvyZCCcLJQXG+3Ng+PXrssvBxRWUgWV1XerLVE/7PUQyw==",
    "475": "eJwVy8EBwCAMw8BVPIAfmCQE3P33Krx1Gt9kcnFTg5pUfsOCPBkIJwvlxUZ78+D4sdvlhwMKK6mCyur7qC1RP+zjEMk=",
    "479": "eJwVy8EBwCAMw8BVPIAfmJAEzP57tbx1GndysbipQU1qUXWHBXkyEF5MpIuN9ubB8ZN/l58PKPyuhNIqqqG2RH0yPRGQ",
    "487": "eJwVyrEBwCAMA7BXfIAHTBIC7v9/FWZpfJPJxU0NalL5DQvyZCCcLJQXG+3Ng+PXrssvBxRWUgWV1Xerfagf3D0Qog==",
    "491": "eJwVy8EBwCAMw8BVPIAfmCQE3P33Krx1Gt9kcnFTg5pUfsOCPBkIJwvlxUZ78+D4sdvlhwMKK6mCyur7qH2oH9wtEKA=",
    "495": "eJwVy8EBwCAMw8BVPIAfmJAEzP57tbx1GndysbipQU1qUXWHBXkyEF5MpIuN9ubB8ZN/l58PKPyuhNIqqqH2oT4gwBFn",
    "499": "eJwVy8EBwCAMw8BVPIAfmCQE3P33Krx1Gt9kcnFTg5pUfsOCPBkIJwvlxUZ78+D4sdvlhwMKK6mCyur7qN3UD9wnEJ4=",
    "503": "eJwVy8EBwCAMw8BVPIAfmJAEzP57tbx1GndysbipQU1qUXWHBXkyEF5MpIuN9ubB8ZN/l58PKPyuhNIqqqF2Ux8guhFl",
    "507": "eJwVy8EBwCAMw8BVPIAfmJAEzP57tbx1GndysbipQU1qUXWHBXkyEF5MpIuN9ubB8ZN/l58PKPyuhNIqqqF2Uh8gtBFj",
    "509": "eJwVy8EBwCAMw8BVPIAfmJAEzP57tbx1GndysbipQU1qUXWHBXkyEF5MpIuN9ubB8ZN/l58PKPyuhNIqqqF2UB8grhFh",
    "511": "eJwVy8EBwCAMw8BVPIAfGAghZv+92rylG29y8/BSg5rUpg5Vb1iQJxeWNwPhw0T6slDu+e9ykwUtNwwo3DyhtIr6AGkaEi4=",
    "1023": "eJwVjMcBwCAMA1fRAHogU4yV/fcKfl8ZX3Dx8FKDCmpRh7oMfcOCHJyYXtzYPkykLwvl9h+Xu5rQdLcb2u5DQun3UUHlEPUDULIUeA=="
}

graph_lru_cache_memo = {}
def graph_lru_cache(func):
    def wrapper(n):
        if n in graph_lru_cache_memo:
            return graph_lru_cache_memo[n].copy_graph()
        g = func(n)
        graph_lru_cache_memo[n] = g
        return g.copy_graph()
    return wrapper

@graph_lru_cache
def construct_1_n(n: int) -> BeltGraph:
    assert isinstance(n, int)
    assert n >= 1
    if n == 1:
        g = BeltGraph()
        g.add_edge(0, 1)
        g.set_input(0)
        g.set_output(1)
        return g
    if n == 2:
        g = BeltGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.set_input(0)
        g.set_output(2)
        g.set_output(3)
        return g
    if n % 2 == 0:
        g = construct_1_n(n//2)
        for i in range(len(g.outputs)-1,-1,-1):
            u = g.disconnect_output(g.outputs[i])
            v = g.new_vertex()
            g.add_edge(u, v)
            w = g.new_vertex()
            g.add_edge(v, w)
            g.set_output(w)
            w = g.new_vertex()
            g.add_edge(v, w)
            g.set_output(w)
        return g
    
    g = BeltGraph()
    g.load_compact_string(_1ns[str(n)])
    # Fix up outputs
    ev = g.evaluate()
    output_flow = ev['output_flow']
    to_combine = {}
    for c, f in output_flow.items():
        f = f[g.inputs[0]]
        if f.denominator > n:
            to_combine[c] = f.denominator // n
    tci = list(to_combine.items())
    if len(tci) > 0:
        while len(tci) > 0:
            options = []
            for i in range(len(tci) - 1):
                c1, r1 = tci[i]
                for j in range(i + 1, len(tci)):
                    c2, r2 = tci[j]
                    _gcd = gcd(r1, r2)
                    if (c1 * r2 + c2 * r1) % _gcd == 0:
                        options.append((c1, c2, i, j))
            c1, c2, i, j = max(options, key=lambda t: t[0]*to_combine[t[1]] + t[1]*to_combine[t[0]])
            del tci[j], tci[i]
            u = g.disconnect_output(c1)
            v = g.new_vertex()
            g.add_edge(u, v)
            u = g.disconnect_output(c2)
            g.add_edge_or_combine(u, v)
            w = g.new_vertex()
            g.add_edge(v, w)
            g.set_output(w)
        ev = g.evaluate()
        output_flow = ev['output_flow']
    for i in range(len(g.outputs)-1,-1,-1):
        u = g.outputs[i]
        f = output_flow[u][g.inputs[0]]
        num, den = f.numerator, f.denominator
        if n < den:
            continue
        if num == 1 and den == n:
            continue
        if den != n:
            num *= n // den
        g2 = construct_1_n(num)
        f, t = g.insert_graph(g2)
        v = g.disconnect_output(u)
        g.add_edge(v, f[0])
        for u in t:
            v = g.new_vertex()
            g.add_edge(u, v)
            g.set_output(v)
    return g

@graph_lru_cache
def construct_n_n(n: int) -> BeltGraph:
    assert isinstance(n, int)
    assert n >= 1
    if n == 1:
        g = BeltGraph()
        g.add_edge(0, 1)
        g.set_input(0)
        g.set_output(1)
        return g
    if n == 2:
        g = BeltGraph()
        g.add_edge(0, 2)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(2, 4)
        g.set_input(0)
        g.set_input(1)
        g.set_output(3)
        g.set_output(4)
        return g
    if n % 2 == 0:
        g = construct_n_n(n//2)
        g = g.doubled()
        return g
    g = construct_1_n(n)
    b = construct_n_n(n-1)
    k = []
    for v in g.vertices():
        if g.is_output(v): continue
        if g.in_degree(v) == 1:
            k.append(v)
    if len(k) != n-1:
        raise Exception(f'Cannot create a {n}-{n}')
    f, t = g.insert_graph(b)
    for v in f:
        u = g.new_vertex()
        g.set_input(u)
        g.add_edge(u, v)
    for u, v in zip(t, k):
        g.add_edge(u, v)
    return g

@graph_lru_cache
def construct_m_n_fast(m: int, n: int) -> BeltGraph:
    if m == 1:
        return construct_1_n(n)
    if m == n:
        return construct_n_n(n)
    if m % 2 == 0 and n % 2 == 0:
        print(f'{m}-{n} | Doubling from {m//2}-{n//2}')
        g = construct_m_n_fast(m//2, n//2)
        g = g.rearrange_vertices_by_depth()
        print(f'{m}-{n} | Before double')
        print(g)
        g.simplify()
        print(f'{m}-{n} | Before double (simplified)')
        print(g)
        print(f'{m}-{n} | Got {g.summary}')
        g = g.doubled()
        print(f'{m}-{n} | After double')
        print(g)
        return g
    if m % 2 == 1 and n % 2 == 1:
        g = construct_m_n_fast(m+1, n+1)
        u = g.disconnect_output(g.outputs[-1])
        v = g.disconnect_input(g.inputs[-1])
        g.add_edge_or_combine(u, v)
        return g

    g = construct_1_n(n)

    # Now convert from 1-n to m-n using m-(n-1)

    g.disconnect_input(g.inputs[0])
    print(f'{m}-{n} | Constructing {m}-{n-1}')
    b = construct_m_n_fast(m, n-1)
    print(f'{m}-{n} | Using {b.summary}')
    k = []
    for v in g.vertices():
        if g.is_output(v): continue
        if g.in_degree(v) == 1:
            k.append(v)
    f, t = g.insert_graph(b)
    for v in f:
        u = g.new_vertex()
        g.set_input(u)
        g.add_edge(u, v)
    for u, v in zip(t, k):
        g.add_edge(u, v)
    print(f'{m}-{n} | After combination')
    print(g)
    return g

if __name__ == '__main__':
    # g = construct_m_n_fast(6, 8)
    for n in range(1, 233):
        try:
            g = construct_n_n(n)
        except Exception:
            continue
        g.simplify()
        g = g.rearrange_vertices_by_depth()
        print(g.summary)
        print(g.is_solved())


