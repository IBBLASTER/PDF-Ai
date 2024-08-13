css = """
<style>
#user, #bot {
    width: 44rem;
    max-height: 150rem;
    background-color: rgb(59 59 67);
    padding: 1rem;
    border-radius: 0.5rem;
    display: flex;
    margin-bottom: 1rem;
}

#bot {
    background-color: rgb(27 25 41);
}

img {
    width: 3rem;
    height: 3rem;
    border-radius: 50%;
    border: 2px solid grey;
    margin-right: 1rem;
}

.msg {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 1rem;
    width: calc(100% - 3rem);
    margin-top: 0.6rem;
</style>
"""

bot = """
<div id="bot">
    <div class="img"><img src="https://cdn.leonardo.ai/users/5cd0a735-8de9-4392-aee0-a38dd75469e5/generations/06899d3a-2381-4706-8aa6-aa28334b5d2a/Default_Design_a_captivating_logo_for_a_cuttingedge_app_that_h_2.jpg" alt=""></div>
    <div class="msg">{{MSG}}</div>
</div>
"""

user = """
<div id="user">
    <div class="img"><img src="https://icons.veryicon.com/png/o/business/multi-color-financial-and-business-icons/user-139.png" alt=""></div>
    <div class="msg">{{MSG}}</div>
</div>
"""
