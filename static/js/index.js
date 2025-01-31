(function(){
  
  var chat = {
    messageToSend: '',
    init: function() {
      this.cacheDOM();
      this.bindEvents();
      this.render();
    },
    cacheDOM: function() {
      this.$chatHistory = $('.chat-history');
      this.$button = $('button');
      this.$textarea = $('#message-to-send');
      this.$chatHistoryList =  this.$chatHistory.find('ul');
    },
    bindEvents: function() {
      this.$button.on('click', this.addMessage.bind(this));
      this.$textarea.on('keyup', this.addMessageEnter.bind(this));
    },
    render: function() {
      this.scrollToBottom();
      if (this.messageToSend.trim() !== '') {
        var template = Handlebars.compile( $("#message-template").html());
        var context = { 
          messageOutput: this.messageToSend,
          time: this.getCurrentTime()
        };

        this.$chatHistoryList.append(template(context));
        this.scrollToBottom();
        this.$textarea.val('');
        
        // responses
        let instance = this;
        $.ajax({
          url: '/query',
          method: 'post',
          contentType: 'application/json; charset=UTF-8',
          data: JSON.stringify({
            recipient: 'wikipedia-en',
            question: this.messageToSend.trim()
          }),
          dataType: 'json'
        }).done(function(answers) {
          console.log(answers)
          if (answers.length == 0) {
            return
          }

          answer = answers[0]['span']

          text = answers[0]['text']
          start = answers[0]['start']
          end = answers[0]['end']

          title = text.split(".")[0].trim()
          title_path = title.split(" ").join("_")

          before_answer = text.slice(0, start) 
          body_answer = text.slice(start, end)
          after_answer = text.slice(end)

          var templateResponse = Handlebars.compile($("#message-response-template").html());
          response = '<strong>' + answers[0]['span'] + 
              '</strong> from <a href="https://en.wikipedia.org/wiki/' +
              answers[0]['doc_id'] + '">' + answers[0]['doc_id'] + '</a>';
          
          if (answer !== "") {
            var contextResponse = {
              // answer: answers[0]['span'],
              // doc_id: answers[0]['doc_id'],
              before_answer: before_answer,
              body_answer: body_answer,
              after_answer: after_answer,
              title: title,
              title_path: title_path,
              time: instance.getCurrentTime()
            };
          } else {
            var contextResponse = {
              // answer: answers[0]['span'],
              // doc_id: answers[0]['doc_id'],
              before_answer: "",
              body_answer: "Could not find any answer for that question!",
              after_answer: "",
              title: "",
              title_path: "",
              time: instance.getCurrentTime()
            };
          }
          instance.$chatHistoryList.append(templateResponse(contextResponse));
          instance.scrollToBottom();
        });
      }
    },
    
    addMessage: function() {
      this.messageToSend = this.$textarea.val()
      this.render();         
    },
    addMessageEnter: function(event) {
        // enter was pressed
        if (event.keyCode === 13) {
          this.addMessage();
        }
    },
    scrollToBottom: function() {
       this.$chatHistory.scrollTop(this.$chatHistory[0].scrollHeight);
    },
    getCurrentTime: function() {
      return new Date().toLocaleTimeString().
              replace(/([\d]+:[\d]{2})(:[\d]{2})(.*)/, "$1$3");
    },
    getRandomItem: function(arr) {
      return arr[Math.floor(Math.random()*arr.length)];
    }
    
  };
  
  chat.init();
  
  var searchFilter = {
    options: { valueNames: ['name'] },
    init: function() {
      var userList = new List('people-list', this.options);
      var noItems = $('<li id="no-items-found">No items found</li>');
      
      userList.on('updated', function(list) {
        if (list.matchingItems.length === 0) {
          $(list.list).append(noItems);
        } else {
          noItems.detach();
        }
      });
    }
  };
  
  searchFilter.init();
  
})();