var TextToSpeech = function (div) {
    'use strict';
    var tts = this;
    tts.textPub = new window.ros.Topic({
        name: 'wt_speech',
        messageType: 'std_msgs/String'
    });
    tts.textPub.advertise();
    tts.say = function (text) {
        var msg = new window.ros.Message({data:text});
        tts.textPub.publish(msg);
        log('Saying: \"' + text + ' \"');
    };
}

function initTTS(ttsTabId){
    window.tts = new TextToSpeech();
    var phrases = ['Yes',
                    'No',
                    'Maybe',
                    'OK',
                    'k',
                    'Board Please',
                    'I don\'t know',
                    'I don\'t care',
                    'One Moment',
                    'Please',
                    'Thank You',
                    'I agree',
                    'I disagree',
                    'A lot',
                    'A little',
                    'Hello',
                    'Goodbye'];
    $('#'+ttsTabId).append('<table><tr><td id="tts_r1c1"></td><td id=tts_r1c2></td></tr>');
    $('#'+ttsTabId).append('<tr><td id="tts_r2c1"></td><td id="tts_r2c2"></td></tr></table>');
    $('#tts_r1c1').append('<button id="submit_text" type="button">Speak:</button>');
    $('#tts_r1c2').append('<input id="txt2say" type="text" style="height:auto;width:105%"/>');
    $('#tts_r2c1').append('Phrases:');
    $('#tts_r2c2').append('<select id="tts_select"></select>');
    for (var i=0; i<phrases.length; i++) {
        $('#tts_select').append('<option value="'+phrases[i]+'">'+phrases[i]+'</option>');
    };
    $("#tts_select").change(function () {
        window.tts.say($(this).val());
        $('#txt2say').val($(this).val());
    });
    $('#submit_text').click(function(){
        window.tts.say($('#txt2say').val());
    });
    $("#txt2say").keyboard({layout:'qwerty', stayOpen:true});
}
